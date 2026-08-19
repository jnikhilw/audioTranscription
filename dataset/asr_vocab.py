# Character-level ASR vocabulary containing the CTC blank token,
# lowercase alphabet characters, space, and apostrophe.
VOCAB = [
    "<blank>",
    "a", "b", "c", "d", "e", "f", "g",
    "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    " ",
    "'",
]


CHAR_TO_ID = {char: idx for idx, char in enumerate(VOCAB)}
ID_TO_CHAR = {idx: char for idx, char in enumerate(VOCAB)}


BLANK_ID = CHAR_TO_ID["<blank>"]
VOCAB_SIZE = len(VOCAB)
