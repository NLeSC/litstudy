import re
import io
import locale
from codecs import BOM_UTF8, BOM_UTF16_BE, BOM_UTF16_LE
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


def canonical(key, aggresive=True, stopwords=None):
    if stopwords is None:
        stopwords = STOPWORDS

    if aggresive:
        key = unidecode(key).lower()

    tokens = re.split(r'[\W]+', key)
    new_tokens = []

    for token in tokens:
        if not token or token[0].isdigit():
            continue

        if aggresive and (token in stopwords or len(token) <= 1):
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


def robust_open(path, errors="replace"):
    """ This function can be used as a drop-in replacement when using
    `with open(path) as f:` to read a file. However, the normal `open` function
    is fragile since it attempts to open the file using the default system
    character encoding and fails immediately when a character cannot be
    decoded. This function is more robust in that it attempts to figure out
    the encoding of the given file and ignores decoding errors.
    """
    if hasattr(path, "read"):
        return path
    elif isinstance(path, bytes):
        content = path
    else:
        with open(path, "rb") as f:
            content = f.read()

    # use the following options:
    # - UTF-8 BOM: decode as UTF-8
    # - UTF-16 BE BOM: decode as UTF-16-BE
    # - UTF-16 LE BOM: decode as UTF-16-LE
    # - otherwise, decode as utf-8 with strict errors
    # - if that fails, decode using default charset
    # - if that fails, decode using utf-8 but ignore errors
    if content.startswith(BOM_UTF8):
        n = len(BOM_UTF8)
        result = content[n:].decode(errors=errors)
    elif content.startswith(BOM_UTF16_BE):
        n = len(BOM_UTF16_BE)
        result = content[n:].decode("utf_16_be", errors=errors)
    elif content.startswith(BOM_UTF16_LE):
        n = len(BOM_UTF16_LE)
        result = content[n:].decode("utf_16_le", errors=errors)
    else:
        try:
            result = content.decode("utf-8", errors="strict")
        except UnicodeError:
            try:
                default_charset = locale.getpreferredencoding()
                result = content.decode(default_charset, errors=errors)
            except UnicodeError:
                result = content.decode("utf-8", errors=errors)

    return io.StringIO(result)
