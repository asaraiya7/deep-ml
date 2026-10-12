import string
import re
import numpy as np

PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def normalize(text: str) -> str:
    text = text.lower().translate(PUNCT_TABLE)
    return " ".join(text.split())
    
def exact_match_score(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if not predictions:
        return 0.0

    return sum(
        normalize(pred) == normalize(ref)
        for pred, ref in zip(predictions, references)
    ) / len(predictions)
