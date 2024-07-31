import csv
import datetime

from ..types import Author, Document, DocumentSet, DocumentIdentifier
from ..common import robust_open, fuzzy_match


class CsvAuthor(Author):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class CsvDocument(Document):
    def __init__(self, record, fields):
        self.fields = fields
        self.record = record

        id = DocumentIdentifier(self.title, doi=self._field("doi"), pubmed=self._field("pubmed"))
        super().__init__(id)

    def __getitem__(self, key):
        return self.record[key]

    def __setitem__(self, key, value):
        self.record[key] = value

    def __iter__(self):
        return iter(self.record)

    def _field(self, field_name):
        key = self.fields[field_name]
        return self.record.get(key) or None

    @property
    def title(self):
        return self._field("title")

    @property
    def abstract(self):
        return self._field("abstract")

    @property
    def publication_source(self):
        return self._field("source")

    @property
    def language(self):
        return self._field("language")

    @property
    def publisher(self):
        return self._field("publisher")

    @property
    def citation_count(self):
        try:
            return int(self._field("citation"))
        except Exception:
            return None

    @property
    def keywords(self):
        text = self._field("keywords")
        if not text:
            return None

        # Try to split on something
        for delim in ";|\t, ":
            if delim in text:
                return [t.strip() for t in text.split(delim)]

        return [text]

    @property
    def publication_date(self):
        text = self._field("date")
        if not text:
            return None

        # Is it a year?
        try:
            year = int(text)
            if year > 1500 and year < 2500:
                return datetime.date(year, 1, 1)
            else:
                return None
        except Exception:
            pass

        # Is it an iso date?
        try:
            return datetime.date.fromisoformat(text)
        except Exception:
            pass

        # Is it one of these formats?
        formats = [
            "%c",
            "%x",
            "%d/%m/%y",
            "%d/%m/%Y",
            "%m/%d/%y",
            "%m/%d/%Y",
            "%d.%m.%y",
            "%d.%m.%Y",
            "%Y-%m-%d",
            "%y-%m-%d",
            "%Y-%d",
            "%d-%Y",
            "%y-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                pass

        # I give up, failed to parse date
        return None

    @property
    def publication_year(self):
        date = self.publication_date
        if not date:
            return None

        return date.year

    @property
    def authors(self):
        text = self._field("authors")
        if not text:
            return None

        for delim in [";", "|", " and ", ","]:
            if delim in text:
                names = text.split(delim)
                names = [name.strip() for name in names]
                names = [name for name in names if name]
                return [CsvAuthor(name) for name in names]

        # Just one author?
        return [CsvAuthor(text)]


def find_field(columns, possible_names):
    PREFIXES = ["", "document", "article", "paper", "item", "publication"]

    for a in possible_names:
        for b in columns:
            for prefix in PREFIXES:
                if fuzzy_match(f"{prefix} {a}", b):
                    return b


def load_csv(
    path: str,
    dialect: "csv.Dialect" = None,
    title_field: str = None,
    authors_field: str = None,
    abstract_field: str = None,
    citation_field: str = None,
    date_field: str = None,
    source_field: str = None,
    doi_field: str = None,
    filter=None,
) -> DocumentSet:
    """Load an abitrary CSV file and parse its contents as a ``DocumentSet``
    on a best effort basis.

    An attempt is made to guess the purpose of the fields of the CSV file
    based on their names. For example, the date of publication is likely
    given by a field named something like "Publication Date",
    "Year of Publication", or "Published Year". In case the field name
    cannot be determined, it is possible to explicitly set the purpose of
    field names by passing additional parameters. For example, ``date_field``
    explicit sets name of the date field.

    The CSV is parsed using the given ``dialect``. If not dialect is given, an
    attempt is made to guess the dialect based on the file's content.

    :param path: Name of CSV file.
    :param dialect: Used to read the CSV file.
    :param title_field: Field name for ``title``.
    :param authors_field: Field name for ``authors``.
    :param abstract_field: Field name for ``abstract``.
    :param citation_field: Field name for ``citation_count``.
    :param date_field: Field name for ``publication_date`` or
    :param source_field: Field name for ``source``.
    :param doi_field: Field name for ``doi``.
    :param filter: Optional function applied to each loaded record. This
                   function can be used to, for example, add or delete fields.

    Example::

        docs = litstudy.load_csv("my_data.csv",
                                 title_field="Document Title",
                                 date_field="Pub Date")
    """
    with robust_open(path) as f:
        text = f.read()

        # If file is empty, exit now
        if not text:
            return DocumentSet([])

        # Guess CSV dialect
        if dialect is None:
            dialect = csv.Sniffer().sniff(text)
            f.seek(0)

        # Read the records
        records = []
        for record in csv.DictReader(f):
            if filter:
                record = filter(record)

            if record:
                records.append(record)

    # No records, exit now
    if not records:
        return DocumentSet([])

    # Get the colum names
    columns = list(records[0].keys())

    # Guess the field names
    fields = dict(
        title=title_field
        or find_field(
            columns,
            [
                "title",
            ],
        ),
        authors=authors_field
        or find_field(
            columns,
            [
                "authors",
                "author(s)",
                "author",
                "names",
                "people",
                "person",
                "persons",
            ],
        ),
        abstract=abstract_field
        or find_field(
            columns,
            [
                "abstract",
                "description",
                "content",
                "text",
                "short text",
                "body",
            ],
        ),
        citation=citation_field
        or find_field(
            columns,
            [
                "citation count",
                "citations count",
                "number of citations",
                "number citations",
                "cited by",
                "citations",
                "cited",
            ],
        ),
        date=date_field
        or find_field(
            columns,
            [
                "pub date",
                "datum",
                "date of publication",
                "published date",
                "publishing date",
                "pub year",
                "year of publication",
                "published year",
                "publishing year",
                "date",
                "year",
            ],
        ),
        source=source_field
        or find_field(
            columns,
            [
                "source title",
                "source name",
                "source",
                "conference name",
            ],
        ),
        pubmed=find_field(
            columns,
            [
                "pubmed",
                "pubmedid",
                "pubmed id",
            ],
        ),
        doi=doi_field
        or find_field(
            columns,
            [
                "doi",
                "object identifier",
                "object identification",
            ],
        ),
        keywords=find_field(
            columns,
            [
                "keywords",
                "tags",
                "categories",
                "keys",
                "indices",
                "author keywords",
                "author tags",
            ],
        ),
        publisher=find_field(
            columns,
            [
                "publisher",
                "publisher name",
            ],
        ),
        language=find_field(
            columns,
            [
                "language",
                "lang",
                "original language",
            ],
        ),
    )

    docs = [CsvDocument(record, fields) for record in records]

    return DocumentSet(docs)
