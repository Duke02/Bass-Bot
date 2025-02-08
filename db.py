from pathlib import Path
import sqlite3
import typing as tp


def does_table_exist(db: sqlite3.Connection, table_name: str) -> bool:
    cursor = db.cursor()
    result = cursor.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?', (table_name,))
    return result.fetchone() is not None


class FeedbackDatabase:
    def __init__(self):
        self.db_path: Path = Path('.').resolve() / 'data' / 'feedback.db'

        if not self.db_path.exists():
            raise FileNotFoundError(f'{self.db_path} does not exist even though volumes should be being used.')

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        if not does_table_exist(self.conn, 'feedback'):
            # message_id must be Bass Bot's reply to the user's prompt,
            self.cursor.execute('CREATE TABLE feedback(row_id int PRIMARY KEY, message_id int NOT NULL, user_id int NOT NULL, one int, two int)')
            self.conn.commit()

        if not does_table_exist(self.conn, 'prompts'):
            # message_id must be Bass Bot's reply to the user's prompt, so therefore the filepath must be a valid path that exists.
            self.cursor.execute('CREATE TABLE prompts(message_id int PRIMARY KEY, prompt VARCHAR(512) NOT NULL, one_filepath VARCHAR(512) NOT NULL, two_filepath VARCHAR(512) NOT NULL)')
            self.conn.commit()

    def create_prompt(self, message_id: int, prompt: str, one_sample_path: Path, two_sample_path: Path):
        if not one_sample_path.exists():
            raise FileNotFoundError(f'Music sample path {one_sample_path} does not exist!')
        if not two_sample_path.exists():
            raise FileNotFoundError(f'Music sample path {two_sample_path} does not exist!')
        self.cursor.execute('INSERT INTO prompts(message_id, prompt, one_filepath, two_filepath) VALUES (?, ?, ?, ?)', (message_id, prompt[:512], str(one_sample_path)[:512], str(two_sample_path)[:512]))
        self.conn.commit()

    def create_new_feedback(self, message_id: int, user_id: int):
        self.cursor.execute('INSERT INTO feedback(message_id, user_id, one, two) VALUES (?, ?, 0, 0)', (message_id, user_id))
        self.conn.commit()

    def contains_message(self, message_id: int) -> bool:
        self.cursor.execute('SELECT user_id FROM feedback WHERE message_id = ?', (message_id,))
        return self.cursor.fetchone() is not None

    def update_feedback(self, message_id: int, user_id: int, is_one_preferred: bool):
        update_field: str = 'one' if is_one_preferred else 'two'
        zero_field: str = 'two' if is_one_preferred else 'one'
        self.cursor.execute(f'UPDATE feedback SET {update_field} = 1, {zero_field} = 0 WHERE message_id = ? AND user_id = ?', (message_id, user_id))
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
