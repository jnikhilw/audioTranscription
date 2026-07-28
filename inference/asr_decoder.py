from pathlib import Path

import torch
from pyctcdecode import build_ctcdecoder

from dataset.asr_vocab import ID_TO_CHAR, VOCAB


BEAM_WIDTH = 50

DECODER_LABELS = list(VOCAB)
DECODER_LABELS[0] = ""

PROJECT_ROOT = Path(__file__).resolve().parent.parent

LM_DIR = (
    PROJECT_ROOT
    / "language_models"
    / "speechtotext_en_us_lm_vdeployable_v4.1"
)

LM_PATH = LM_DIR / "3-gram.pruned.3e-7.arpa"

if not LM_PATH.is_file():
    raise FileNotFoundError(
        f"Language model not found: {LM_PATH}"
    )

decoder = build_ctcdecoder(
    labels=DECODER_LABELS,
    kenlm_model_path=str(LM_PATH),
    alpha=0.2,
    beta=0,
)


def ctc_decode_greedy(predicted_ids):
    ids = predicted_ids[0].tolist()

    collapsed = []
    prev = None

    for token in ids:
        if token != prev:
            collapsed.append(token)

        prev = token

    collapsed = [
        token
        for token in collapsed
        if token != 0
    ]

    return "".join(
        ID_TO_CHAR[token]
        for token in collapsed
    )


def ctc_decode_beam(logits):
    probs = (
        torch.softmax(logits[0], dim=-1)
        .detach()
        .cpu()
        .numpy()
    )

    assert probs.shape[1] == len(DECODER_LABELS)

    return decoder.decode(
        probs,
        beam_width=BEAM_WIDTH,
    )