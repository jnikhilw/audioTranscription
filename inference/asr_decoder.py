from dataset.asr_vocab import ID_TO_CHAR, VOCAB
from pyctcdecode import build_ctcdecoder
import numpy as np


decoder = build_ctcdecoder(
    labels=VOCAB[1:]
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
        token for token in collapsed
        if token != 0
    ]

    return "".join(
        ID_TO_CHAR[token]
        for token in collapsed
    )


def ctc_decode_beam(log_probs):

    probs = np.exp(
        log_probs[0]
        .detach()
        .cpu()
        .numpy()
    )

    return decoder.decode(probs)