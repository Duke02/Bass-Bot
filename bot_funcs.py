import typing as tp
from datetime import datetime
from pathlib import Path

import discord
import numpy as np
import scipy
import torch.cuda
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def get_sentiment_runner() -> tp.Callable[[str, str], int]:
    task = 'sentiment'
    MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, device_map=get_device())
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, device_map=get_device())

    def inner(text: str, bot_mention: str) -> int:
        text = text.replace(bot_mention, '').lower().strip()
        print(f'Getting sentiment for "{text}"')
        encoded_input = tokenizer(text, return_tensors='pt').to(get_device())
        output = model(**encoded_input)
        scores: np.ndarray = output[0][0].detach().cpu().numpy()
        scores = softmax(scores)
        rankings: np.ndarray = np.argsort(scores)
        return int(rankings[-1])

    print('Set up sentiment runner')
    return inner


def normalize(sample: np.ndarray, norm_rate: float = 0.9):
    norm_factor: float = np.abs(sample.max()) * norm_rate
    return sample / norm_factor


def low_pass(sample: np.ndarray, cutoff: float, sampling_rate: float) -> np.ndarray:
    sos_out: np.ndarray = scipy.signal.butter(10, cutoff, btype='lowpass', fs=sampling_rate, output='sos')
    return scipy.signal.sosfilt(sos_out, sample)


def high_pass(sample: np.ndarray, cutoff: float, sampling_rate: float) -> np.ndarray:
    sos_out: np.ndarray = scipy.signal.butter(10, cutoff, btype='highpass', fs=sampling_rate, output='sos')
    return scipy.signal.sosfilt(sos_out, sample)


def postprocess_sample(sample: np.ndarray, sampling_rate: tp.Union[int, float], do_norm: bool = False,
                       do_low: bool = False, do_high: bool = False) -> np.ndarray:
    # hyperparameters.
    low_pass_cutoff_freq: float = min(float(sampling_rate) / 2.0, 10_000)
    high_pass_cutoff_freq: float = 100.0

    print(low_pass_cutoff_freq, high_pass_cutoff_freq)

    out: np.ndarray = np.copy(sample)
    print(out.shape)
    if do_norm:
        out = normalize(out)
        print(out.min(), out.max(), out.mean())
    if do_low:
        out = low_pass(out, low_pass_cutoff_freq, sampling_rate)
        print(out.min(), out.max(), out.mean())
    if do_high:
        out = high_pass(out, high_pass_cutoff_freq, sampling_rate)
        print(out.min(), out.max(), out.mean())
    return out


def get_music_runner() -> tp.Callable[[str, tp.Optional[str]], tuple[Path, discord.File]]:
    model: str = 'facebook/musicgen-large'
    synthesiser = pipeline("text-to-audio", model=model, device=get_device())

    def prompt_to_filename(prompt: str, annot: tp.Optional[str]) -> Path:
        curr_time: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        mod: str = '' if annot is None else f'_{annot}_'
        filename: str = curr_time + mod + '_'.join(prompt.lower().split(' ')[:5]) + '.wav'
        data_dir: Path = Path('.').resolve() / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / filename

    def inner(prompt: str, annot: tp.Optional[str]) -> tuple[Path, discord.File]:
        filename: Path = prompt_to_filename(prompt, annot)
        sample = synthesiser(prompt, forward_params={'do_sample': True})
        sampling_rate = sample['sampling_rate']
        # actual_audio: np.ndarray = postprocess_sample(sample['audio'], sampling_rate)
        scipy.io.wavfile.write(filename, rate=sampling_rate, data=sample['audio'])
        return filename, discord.File(filename)

    print('Set up music runner')
    return inner
