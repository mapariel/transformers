import re
from typing import List


def simple_cleansing(text: str) -> str:
    """
    Simple cleaning of the text, mostly lowercase. The cleaning is adapted for
    texts from gutenberg https://www.gutenberg.org/

    Args:
        text: The text to clean

    Returns:
        The cleaned text
    """
    # Remove illustration and copyright tags
    text = ' '.join(re.split(r'\[[^\[]*?\]', text, flags=re.DOTALL))
    text = ' '.join(re.split(r'\[[^\[]*?\]', text, flags=re.DOTALL))
    text = text.lower()

    # Remove line starting with chapter
    text = ' '.join(re.split(r'chapter.*', text))

    # Some words are starting or delimited by _
    text = ' '.join(re.split(r'_([a-zéê\-à]*)_', text))
    text = ' '.join(re.split(r'_([a-z]*)', text))
    text = ' '.join(re.split(r'--', text))

    text = ' '.join(text.split())

    # Replace question and exclamation
    # table = " ".maketrans("?!:", "...", "()*/”“’‘,;")
    table = " ".maketrans("", "", "()*/”“’‘")  # remove apostrophe and guillemot
    text = text.translate(table)

    return text


def vectorize(tokens: List[str], vocabulary: List[str]) -> List[int]:
    """
    Transforms a list of tokens (words) in a sequence of indices of the tokens in the vocabulary
    Args:
        tokens: List of strings
        vocabulary: List of strings

    Returns:
        A list of indices (integers)
    """
    vector = [vocabulary.index(token) for token in tokens]
    return vector
