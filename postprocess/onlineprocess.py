
common_mispellings = {
    "dont": "don't",
    "doesnt": "doesn't",
    "didnt": "didn't",
    "cant": "can't",
    "wont": "won't",
    "isnt": "isn't",
    "arent": "aren't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "im": "I'm",
    "ive": "I've",
    "ill": "I'll",
    "id": "I'd",
    "youre": "you're",
    "theyre": "they're",
    "weve": "we've",
    "thats": "that's",
    "whats": "what's",
}


def postprocess_online(text: str) -> str:

    """
    Applies lightweight cleanup to decoded online ASR text.
    The function:
    - trims surrounding whitespace
    - collapses repeated spaces
    - capitalizes the standalone pronoun "i"
    - applies curated contraction corrections
    
    """

    if not text:
        return ""

    words = text.split()

    for index, word in enumerate(words):
        if word == "i":
            words[index] = "I"

        else:
            words[index] = common_mispellings.get(word, word)

    return " ".join(words)

    
    
    
