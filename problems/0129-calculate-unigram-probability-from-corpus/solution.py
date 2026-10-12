import re
from collections import Counter

def unigram_probability(corpus: str, word: str) -> float:
    elements_split = re.split(r'\s+', corpus)
    counts = dict(Counter(elements_split))
    return counts[word] / len(elements_split)


