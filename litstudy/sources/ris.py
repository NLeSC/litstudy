from ..types import Document, Author, DocumentSet, DocumentIdentifier
from ..common import robust_open
import logging


def extract_title(attr):
    return attr.get("TI") or attr.get("T1") or attr.get("T2") or attr.get("T3") or attr.get("TT")


class RISDocument(Document):
    def __init__(self, attr, keywords, authors):
        title = extract_title(attr)
        doi = attr.get("DO") or None

        super().__init__(DocumentIdentifier(title, doi=doi))
        self.attr = attr
        self.keyword_list = keywords
        self.author_list = authors

    @property
    def title(self) -> str:
        return extract_title(self.attr)

    @property
    def authors(self):
        return self.author_list

    @property
    def affiliations(self):
        return None

    @property
    def publisher(self):
        return self.attr.get("PB")

    @property
    def language(self):
        return self.attr.get("LA")

    @property
    def publication_year(self):
        try:
            return int(self.attr["PY"])
        except Exception:
            return None

    @property
    def publication_source(self):
        return None

    @property
    def keywords(self):
        return self.keyword_list

    @property
    def abstract(self):
        return self.attr.get("AB")


class RISAuthor(Author):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


def load_ris_file(path: str) -> DocumentSet:
    """Load the RIS file at the given `path` as a `DocumentSet`."""
    docs = []

    with robust_open(path) as f:
        authors = []
        keywords = []
        attr = dict()

        for line in f:
            line = line.strip()
            if not line:  # ignore empty lines?
                continue

            if len(line) < 5 or line[2:5] != "  -":
                raise Exception(f"invalid RIS line: {line}")

            key = line[:2]
            value = line[5:].strip()

            if key == "ER":
                docs.append(RISDocument(attr, keywords, authors))
                attr = dict()
                authors = []
                keywords = []
            elif key == "KW":
                keywords.append(value)
            elif key in ["A1", "A2", "A3", "A4", "AU"]:
                authors.append(RISAuthor(value))
            elif key in attr:
                logging.warn(
                    f"Tag {key} appears multiple times " + f'("{value}" and "{attr[key]}")'
                )
            else:
                attr[key] = value

    # Last document in RIS file
    if attr:
        docs.append(RISDocument(attr, keywords, authors))

    return DocumentSet(docs)
