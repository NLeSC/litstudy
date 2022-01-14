import re
from unidecode import unidecode

try:
    from tqdm import tqdm

    def progress_bar(it):
        return tqdm(it)
except ImportError:
    def progress_bar(it):
        return it

STOPWORDS = set([
    '',
    'and',
    'at',
    'for',
    'in',
    'into',
    'of',
    'on',
    'onto',
    'over',
    'the',
    'to',
    'ltd',
    'corporation',
    'corp',
])


def canonical(key, aggresive=True):
    if aggresive:
        key = unidecode(key).lower()

    tokens = re.split(r'[\W]+', key)
    new_tokens = []

    for token in tokens:
        if not token or token[0].isdigit():
            continue

        if aggresive and (token in STOPWORDS or len(token) <= 1):
            continue

        new_tokens.append(token)

    return ' '.join(new_tokens)


def fuzzy_match(lhs, rhs):
    if lhs is None or rhs is None:
        return False

    return canonical(lhs) == canonical(rhs)


class FuzzyMatcher:
    def __init__(self, initial=None):
        mapping = dict()
        unmapping = dict()

        if initial is not None:
            for src, dst in initial.items():
                key = canonical(src)
                dst_key = canonical(dst)

                mapping[dst_key] = key
                mapping[key] = key
                unmapping[key] = dst

        self.mapping = mapping
        self.unmapping = unmapping

    def get(self, name):
        key = canonical(name)

        if key in self.mapping:
            return self.unmapping[self.mapping[key]]

        nice_name = canonical(name, False)
        self.mapping[key] = key
        self.unmapping[key] = nice_name
        return nice_name
