from ..types import Document, Author, DocumentSet, DocumentIdentifier, Affiliation
from ..common import robust_open
import csv
import logging


class IEEEDocument(Document):
    def __init__(self, entry):
        doi = entry["DOI"] or None
        title = entry["Document Title"]

        super().__init__(DocumentIdentifier(title, doi=doi))
        self.entry = entry

    @property
    def title(self) -> str:
        return self.entry.get("Document Title")

    @property
    def authors(self):
        authors = self.entry.get("Authors", "").split("; ")
        affs = self.entry.get("Author Affiliations", "").split("; ")

        # Bug fix #55:
        # In some cases, the number of affiliations does not match the number of authors
        # given by the CSV file. Since there is no way of knowing which affiliations belong
        # to which authors, we just ignore all affiliations in this case.
        if len(authors) != len(affs):
            logging.warn(
                (
                    f"affiliations for entry '{self.title}' are invalid: the number of authors "
                    f"({len(authors)}) does not match the number of author affilications ({len(affs)})"
                )
            )

            affs = [None] * len(authors)

        return [IEEEAuthor(a, b) for a, b in zip(authors, affs)]

    @property
    def affiliations(self):
        affs = self.entry.get("Author Affiliations", "").split("; ")
        return [IEEEAffiliation(a) for a in affs]

    @property
    def publisher(self):
        return self.entry.get("Publisher")

    @property
    def publication_year(self):
        try:
            return int(self.entry["Publication Year"])
        except Exception:
            return None

    @property
    def keywords(self):
        # There are several sources of keywords. Should we only include
        # author keywords maybe? Or all possible keywords?
        keys = [
            "Author Keywords",
            "IEEE Terms",
            "INSPEC Controlled Terms",
            "INSPEC Non-Controlled Terms",
            "Mesh_Terms",
        ]

        keywords = []
        for key in keys:
            for word in self.entry.get(key, "").split(";"):
                if word:
                    keywords.append(word)

        return keywords

    @property
    def abstract(self):
        return self.entry.get("Abstract") or None

    @property
    def citation_count(self):
        try:
            return int(self.entry["Article Citation Count"])
        except Exception:
            return None


class IEEEAffiliation(Affiliation):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class IEEEAuthor(Author):
    def __init__(self, name, affiliation):
        self._name = name
        self._affiliation = affiliation

    @property
    def name(self):
        return self._name

    @property
    def affiliations(self):
        # Handle special case where affiliation is NA (not applicable)
        if not self._affiliation or self._affiliation == "NA":
            return None

        return [IEEEAffiliation(self._affiliation)]


def load_ieee_csv(path: str) -> DocumentSet:
    """Import CSV file exported from
    `IEEE Xplore <https://ieeexplore.ieee.org/search/searchresult.jsp>`_.
    """
    with robust_open(path) as f:
        lines = csv.DictReader(f)
        docs = [IEEEDocument(line) for line in lines]
        return DocumentSet(docs)
