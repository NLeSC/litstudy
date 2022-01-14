import csv

from ..types import Document, DocumentSet, DocumentIdentifier


class SpringerDocument(Document):
    def __init__(self, entry):
        doi = entry['Item DOI'] or None
        title = entry['Item Title']

        super().__init__(DocumentIdentifier(
                title,
                doi=doi
        ))
        self.entry = entry

    @property
    def title(self) -> str:
        return self.entry['Item Title']

    @property
    def authors(self):
        # While Springer does provide an authors field, all names are
        # concatenated into one long string without seperators, making
        # it impossible to seperate them. Too bad.
        # authors = self.entry.get('Authors')
        return None

    @property
    def publisher(self):
        return 'springer'

    @property
    def publication_year(self):
        try:
            return int(self.entry['Publication Year'])
        except Exception:
            return None


def load_springer_csv(path: str) -> DocumentSet:
    """ """
    with open(path, newline='') as f:
        lines = csv.DictReader(f)
        return [SpringerDocument(line) for line in lines]
