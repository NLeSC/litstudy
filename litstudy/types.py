from abc import ABC, abstractmethod
from datetime import date
from typing import Optional, List
import random
from .clean import fuzzy_match


class DocumentSet:
    def __init__(self, docs):
        self.docs = docs
        """ List of documents as `Document` objects"""

    def filter(self, predicate):
        """Returns a new `DocumentSet` which contains only the documents for
        which the given predicate returned true."""
        return DocumentSet([d for d in self.docs if predicate(d)])

    def difference(self, other):
        haystack = [d.id for d in other]
        result = []

        for doc in self.docs:
            needle = doc.id
            if not any(needle.matches(i) for i in haystack):
                result.append(doc)

        return DocumentSet(result)

    def intersect(self, other):
        result = []

        for a in self.docs:
            found = False

            for b in other.docs:
                if a.id.matches(b.id):
                    found = True
                    break

            if found:
                result.append(a)

        return result

    def union(self, other):
        return DocumentSet(self.docs + other.docs).unique()

    def unique(self):
        result = []

        for doc in self.docs:
            found = False
            needle = doc.id

            for other in result:
                if other.id.matches(needle):
                    other._identifier = other._identifier.merge(needle)
                    found = True

            if not found:
                result.append(doc)

        return DocumentSet(result)

    def sample(self, n, seed=0):
        docs = self.docs

        if len(docs) > n:
            random.seed(seed)
            docs = random.sample(docs, n)

        return DocumentSet(docs)

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersect(other)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, key):
        return self.docs[key]

    def __iter__(self):
        return iter(self.docs)

    def __repr__(self):
        return repr(self.docs)


class DocumentIdentifier:
    def __init__(self, title, **attr):
        # Remove keys where value is None
        self._title = title
        self._attr = dict((k, v) for k, v in attr.items() if v)

    @property
    def title(self) -> Optional[str]:
        return self._title

    @property
    def doi(self) -> Optional[str]:
        return self._attr.get('doi')

    @property
    def isbn(self):
        return self._attr.get('isbn')

    @property
    def eissn(self):
        return self._attr.get('eissn')

    @property
    def pubmmed(self):
        return self._attr.get('pubmed')

    @property
    def eid(self):
        return self._attr.get('eid')

    def matches(self, other):
        n = 0

        # Two identifiers match if all keys that they have in common are equal
        for key in self._attr:
            if key in other._attr:
                if self._attr[key] != other._attr[key]:
                    return False
                n += 1

        return n > 0 or fuzzy_match(self._title, other._title)

    def merge(self, other) -> 'DocumentIdentifier':
        attr = dict()
        attr.update(other._attr)
        attr.update(self._attr)
        return DocumentIdentifier(self._title, **attr)

    def __repr__(self):
        return f'<{self.title}>'


class Document(ABC):
    def __init__(self, identifier: DocumentIdentifier):
        self._identifier = identifier

    @property
    def id(self) -> DocumentIdentifier:
        return self._identifier

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @property
    @abstractmethod
    def authors(self):
        pass

    @property
    def affiliations(self):
        authors = self.authors

        if authors is not None:
            items = dict()

            for author in authors:
                for aff in author.affiliations:
                    items[aff.name] = aff

            return list(items.values())

        return None

    @property
    def publisher(self):
        return None

    @property
    def language(self):
        return None

    @property
    def publication_date(self) -> Optional[date]:
        return None

    @property
    def publication_year(self):
        date = self.publication_date
        if date is None:
            return None
        return date.year

    @property
    def publication_source(self):
        return None

    @property
    def keywords(self):
        return None

    @property
    def abstract(self):
        return None

    @property
    def citation_count(self) -> Optional[int]:
        return None

    @property
    def references(self) -> Optional[List[DocumentIdentifier]]:
        return None


class Affiliation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def city(self) -> Optional[str]:
        pass

    @property
    def country(self) -> Optional[str]:
        pass


class Author(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    def orcid(self):
        return None

    @property
    def affiliations(self) -> 'Optional[list[Affiliation]]':
        return None

