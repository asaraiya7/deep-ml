import string
import re
import numpy as np

def normalize(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")

    if not predictions:
        return 0.0

    return sum(
        normalize(pred) == normalize(ref)
        for pred, ref in zip(predictions, references)
    ) / len(predictions)
