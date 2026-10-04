import argparse

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

DEFAULT_MODEL = "openai/whisper-tiny"
TARGET_SR = 16000


def load_audio(path: str) -> dict:
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        duration = len(data) / sr
        data = np.interp(
            np.linspace(0, duration, int(duration * TARGET_SR), endpoint=False),
            np.linspace(0, duration, len(data), endpoint=False),
            data,
        ).astype(np.float32)
    return {"array": data, "sampling_rate": TARGET_SR}


def main():
    parser = argparse.ArgumentParser(description="Распознавание речи из аудиофайла")
    parser.add_argument("audio", help="путь к аудиофайлу")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="модель Whisper, по умолчанию tiny"
    )
    parser.add_argument(
        "--language",
        default=None,
        help="код языка, например en или ru, по умолчанию авто",
    )
    args = parser.parse_args()

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
    )

    generate_kwargs = {"language": args.language} if args.language else {}
    result = pipe(load_audio(args.audio), generate_kwargs=generate_kwargs)
    print(result["text"].strip())


if __name__ == "__main__":
    main()
