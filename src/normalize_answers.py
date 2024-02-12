import regex
import string
from typing import List
import normalize_text 

# Normalization adapted from SQuAD evaluation script https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def remove_articles(text: str) -> str:
    """
    Removes articles ('a', 'an', 'the') from the text.
    """
    return regex.sub(r'\b(a|an|the)\b', ' ', text)

def white_space_fix(text: str) -> str:
    """
    Fixes extra whitespace in the text by collapsing multiple spaces into one.
    """
    return ' '.join(text.split())

def remove_punc(text: str) -> str:
    """
    Removes punctuation from the text and replaces it with a space.
    """
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    return text

def lower(text: str) -> str:
    """
    Converts all characters in the text to lowercase.
    """
    return text.lower()

def normalize_answer(s: str, lowercase: bool = True) -> str:
    """
    Normalizes answers by removing articles, punctuation, fixing whitespace, and optionally converting to lowercase.
    """
    if lowercase:
        s = lower(s)
    s = normalize_text.normalize(s)
    return white_space_fix(remove_articles(remove_punc(s)))


def is_answer_in_text(text: str, answers: List[str]) -> bool:
    """
    Checks if any of the provided answers are present in the given text after normalization.
    """
    for a in answers:
        normalized_answer_lower = normalize_answer(a, lowercase=True)
        normalized_answer = normalize_answer(a, lowercase=False)
        normalized_text = white_space_fix(remove_punc(text))

        if (a in text or 
            normalized_answer_lower in normalized_text or 
            normalized_answer in normalized_text):
            return True
    
    return False
