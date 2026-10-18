import re

def whitespace_tokenize(text: str) -> list[str]:
    """
    Splits `text` into a list of tokens on whitespace (spaces, tabs,
    newlines), collapsing any run of consecutive whitespace into a
    single split point and dropping leading/trailing whitespace
    entirely.
    """
    return text.split()






def char_tokenize(text: str) -> list[str]:
    """
    Splits `text` into a list of its INDIVIDUAL characters, one token
    per character, including whitespace and punctuation.
    """
    return list(text)
