import string
import re
import numpy as np

def normalize(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def exact_match_score(predictions: list[str], references: list[str]) -> float:
    """
    Calculate the exact match score between predictions and references.
    
    Args:
        predictions: List of predicted strings
        references: List of reference (ground truth) strings
    
    Returns:
        Exact match score as a float between 0 and 1
    """
    # Handle empty lists cases 
    if not len(predictions) or not len(references):
        return 0

    # Normalize strings for exact match score calc.  
    preds_np = np.array([normalize(s) for s in predictions])
    refs_np = np.array([normalize(s) for s in references])

    # Ensure exactly same length 
    assert len(predictions) == len(references)

    return np.sum(preds_np == refs_np) / len(references)
