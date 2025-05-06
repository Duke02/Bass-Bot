from pathlib import Path
import sqlite3
import typing as tp

from datasets import Dataset
import numpy as np
import scipy.io.wavfile as wav
from torch import is_inference_mode_enabled
from tqdm import tqdm


def does_table_exist(db: sqlite3.Connection, table_name: str) -> bool:
    cursor = db.cursor()
    result = cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?', (table_name,))
    return result.fetchone() is not None


class FeedbackDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path: Path = db_path.resolve() or Path('.').resolve() / 'data' / 'feedback.db'

        if not self.db_path.exists():
            raise FileNotFoundError(f'{self.db_path} does not exist even though volumes should be being used.')

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if not does_table_exist(self.conn, 'feedback'):
            # message_id must be Bass Bot's reply to the user's prompt,
            self.cursor.execute(
                'CREATE TABLE feedback(row_id int PRIMARY KEY, message_id int NOT NULL, user_id int NOT NULL, one int, two int)')
            self.conn.commit()

        if not does_table_exist(self.conn, 'prompts'):
            # message_id must be Bass Bot's reply to the user's prompt, so therefore the filepath must be a valid path that exists.
            self.cursor.execute(
                'CREATE TABLE prompts(message_id int PRIMARY KEY, prompt VARCHAR(512) NOT NULL, one_filepath VARCHAR(512) NOT NULL, two_filepath VARCHAR(512) NOT NULL)')
            self.conn.commit()

    def create_prompt(self, message_id: int, prompt: str, one_sample_path: Path, two_sample_path: Path):
        if not one_sample_path.exists():
            raise FileNotFoundError(f'Music sample path {one_sample_path} does not exist!')
        if not two_sample_path.exists():
            raise FileNotFoundError(f'Music sample path {two_sample_path} does not exist!')
        self.cursor.execute('INSERT INTO prompts(message_id, prompt, one_filepath, two_filepath) VALUES (?, ?, ?, ?)',
                            (message_id, prompt[:512], str(one_sample_path)[:512], str(two_sample_path)[:512]))
        self.conn.commit()

    def create_new_feedback(self, message_id: int, user_id: int):
        self.cursor.execute('INSERT INTO feedback(message_id, user_id, one, two) VALUES (?, ?, 0, 0)',
                            (message_id, user_id))
        self.conn.commit()

    def contains_message(self, message_id: int) -> bool:
        self.cursor.execute('SELECT user_id FROM feedback WHERE message_id = ?', (message_id,))
        return self.cursor.fetchone() is not None

    def update_feedback(self, message_id: int, user_id: int, is_one_preferred: bool):
        update_field: str = 'one' if is_one_preferred else 'two'
        zero_field: str = 'two' if is_one_preferred else 'one'
        self.cursor.execute(
            f'UPDATE feedback SET {update_field} = 1, {zero_field} = 0 WHERE message_id = ? AND user_id = ?',
            (message_id, user_id))
        self.conn.commit()

    def count(self, message_id: int) -> int:
        return self.cursor.execute('SELECT COUNT(*) FROM feedback WHERE message_id = ?', (message_id,)).fetchone()[0]

    def is_one_preferred(self, message_id: int) -> bool:
        result = self.cursor.execute('SELECT one, two FROM feedback WHERE message_id = ?', (message_id,))
        ones: int = 0
        twos: int = 0
        for one, two in result.fetchall():
            ones += one
            twos += two
        return ones > twos

    def _try_to_make_filepaths_from_dockerfile(self, p: Path) -> Path:
        if not str(p).startswith('/app'):
            return p
        else:
            return self.db_path.parent / p.name

    def get_prompt_and_generated_data(self, message_id: int) -> tuple[str, int, np.ndarray, np.ndarray]:
        prompt: str
        one_filepath: str
        two_filepath: str
        prompt, one_filepath, two_filepath = self.cursor.execute(
            'SELECT prompt, one_filepath, two_filepath FROM prompts WHERE message_id = ?', (message_id,)).fetchone()
        one_filepath: Path = self._try_to_make_filepaths_from_dockerfile(Path(one_filepath))
        two_filepath: Path = self._try_to_make_filepaths_from_dockerfile(Path(two_filepath))

        if not one_filepath.exists():
            raise FileNotFoundError(
                f'Cannot find first generated music sample for message ID of {message_id} ({one_filepath=})')
        if not two_filepath.exists():
            raise FileNotFoundError(
                f'Cannot find second generated music sample for message ID of {message_id} ({two_filepath=})')

        one_sample_rate: int
        two_sample_rate: int
        one_generated: np.ndarray
        two_generated: np.ndarray
        one_sample_rate, one_generated = wav.read(one_filepath)
        two_sample_rate, two_generated = wav.read(two_filepath)

        if one_sample_rate != two_sample_rate:
            raise ValueError(
                f'Generated music samples were not saved with the same sample rate ({one_sample_rate=}, {two_sample_rate=})')

        return prompt, one_sample_rate, one_generated, two_generated

    def get_all_message_ids(self) -> list[int]:
        message_ids: list[int] = [int(row[0]) for row in
                                  self.cursor.execute(f'SELECT message_id from prompts').fetchall()]
        return message_ids

    def to_dpo_dataset(self) -> Dataset:
        message_ids: list[int] = list(set(self.get_all_message_ids()))

        def get_prompt_data(message_id: int) -> dict[str, str | np.ndarray | int] | None:
            try:
                prompt, sample_rate, one, two = self.get_prompt_and_generated_data(message_id)
                return {'prompt': prompt, 'sample_rate': sample_rate, 'one': one.astype(np.float32), 'two': two.astype(np.float32)}
            except FileNotFoundError:
                return None
            except ValueError:
                return None

        def swap_one_two(message_id: int, prompt_d: dict[str, str | np.ndarray | int]) -> dict[str, np.ndarray | str]:
            is_one_preferred: bool = self.is_one_preferred(message_id)
            return {'prompt': prompt_d['prompt'], 'chosen': prompt_d['one'] if is_one_preferred else prompt_d['two'],
                    'rejected': prompt_d['two'] if is_one_preferred else prompt_d['one']}

        prompt_data: dict[int, dict[str, str | np.ndarray | int]] = {message_id: prompt_dict for message_id in
                                                                     tqdm(message_ids, desc='Getting prompt info...') if (prompt_dict := get_prompt_data(
                message_id)) is not None}
        raw_data: dict[int, dict[str, str | np.ndarray]] = {message_id: swap_one_two(message_id, prompt_dict) for
                                                            message_id, prompt_dict in tqdm(prompt_data.items(), desc='Reorganizing prompt data...')}
        actual_dataset_input: dict[str, list[str] | list[np.ndarray]] = {
            'prompt': [p['prompt'] for p in raw_data.values()],
            'chosen': [p['chosen'] for p in raw_data.values()],
            'rejected': [p['rejected'] for p in raw_data.values()]
        }

        return Dataset.from_dict(actual_dataset_input)
