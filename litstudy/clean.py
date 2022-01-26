from collections import Counter
from .common import canonical


def generate_mapping(tokens, stopwords):
    stopwords = set(stopwords)
    mapping = dict()
    result = dict()

    for token, _count in Counter(tokens).most_common():
        key = canonical(token, aggresive=True, stopwords=stopwords)

        if key not in mapping:
            mapping[key] = token
        else:
            result[token] = mapping[key]

    return result
