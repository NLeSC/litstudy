from .types import Document, DocumentSet, DocumentIdentifier, Author
from bibtexparser.customization import convert_to_unicode
import bibtexparser
from datetime import date
import re

MONTHS = dict(
        jan=1, january=1,
        feb=2, feburary=2,
        mar=3, march=3,
        apr=4, april=4,
        may=5,
        jun=6, june=6,
        jul=7, july=7,
        aug=8, august=8,
        sep=9, september=9,
        oct=10, october=10,
        nov=11, november=11,
        dec=12, december=12,
)


class BibDocument(Document):
    def __init__(self, entry):
        title = entry.get('title')
        attr = dict(
                doi=entry.get('doi'),
                isbn=entry.get('isbn'),
                pubmed=entry.get('pmid'),
        )

        super().__init__(DocumentIdentifier(title, **attr))
        self.entry = entry

    @property
    def key(self) -> str:
        return self.entry['ID']

    @property
    def title(self) -> str:
        return self.entry.get('title').strip('{}')

    @property
    def authors(self):
        content = self.entry.get('author')
        if content is None:
            return None

        content = re.sub('[ \r\n\t]+', ' ', content)
        names = re.split(' (and|And|AND) ', content)

        if not names:
            return None

        # remove "other" from the list
        if names[-1] == 'others':
            names = names[:-1]

        return [BibAuthor(name.strip('{}"')) for name in names]

    @property
    def publisher(self):
        return self.entry.get('publisher')

    @property
    def language(self):
        return self.entry.get('language')

    @property
    def publication_date(self):
        year = self.publication_year
        month = self.publication_month or 1

        if year is None:
            return None

        return date(year, month, 1)

    @property
    def publication_year(self):
        try:
            year = int(self.entry['year'])
        except Exception:
            return None

        # Sometimes year is in 19xx format.
        if year < 100:
            year += 1900

        return year

    @property
    def publication_month(self):
        key = self.entry.get('month', '').lower()
        return MONTHS.get(key)  # name -> number

    @property
    def publication_source(self):
        if 'journal' in self.entry:
            return self.entry['journal']
        elif 'booktitle' in self.entry:
            return self.entry['booktitle']

        return None

    @property
    def keywords(self):
        content = self.entry.get('keywords')
        if not content:
            return None

        return [w.strip().lower() for w in re.split('[;,\n\r]+', content) if w]

    @property
    def abstract(self):
        return self.entry.get('abstract')


class BibAuthor(Author):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'<{self._name}>'


def load_bibtex(path):
    parser = bibtexparser.bparser.BibTexParser(common_strings=True)
    # parser.customization = convert_to_unicode

    with open(path) as f:
        data = bibtexparser.load(f, parser=parser)

    docs = [BibDocument(e) for e in data.entries if e.get('title')]
    return DocumentSet(docs)
