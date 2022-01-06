from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from typing import Optional, List
import pandas as pd
import random
import re

from .common import fuzzy_match, canonical, progress_bar


class DocumentSet:
    def __init__(self, docs, data=None):
        if data is None:
            data = pd.DataFrame(index=range(len(docs)))
        else:
            data = pd.DataFrame(data)

        assert len(data) == len(docs)
        self.data = data
        self.docs = docs

    def _refine_docs(self, callback):
        flags = []
        new_docs = []
        old_docs = []

        for i, doc in enumerate(progress_bar(self.docs)):
            new_doc = callback(doc)
            flags.append(new_doc is not None)

            if new_doc is not None:
                new_docs.append(new_doc)
            else:
                old_docs.append(doc)

        new_data = self.data.iloc[flags]
        old_data = self.data.iloc[[not f for f in flags]]

        return DocumentSet(new_docs, new_data), DocumentSet(old_docs, old_data)

    def add_column(self, name: str, values) -> DocumentSet:
        assert len(values) == len(self.docs)
        data = self.data.copy(deep=False)
        data[name] = values
        return DocumentSet(self.docs, data)

    def remove_column(self, name: str) -> DocumentSet:
        data = self.data.copy(deep=False)
        data.drop(name)
        return DocumentSet(self.docs, data)

    def filter_docs(self, predicate) -> DocumentSet:
        return self.filter_meta(lambda doc, _: predicate(doc))

    def filter(self, predicate) -> DocumentSet:
        indices = []

        for doc, record in zip(self.docs, self.data.itertuples()):
            if predicate(doc, record):
                indices.append(record.Index)

        return self.select(indices)

    def select(self, indices) -> DocumentSet:
        data = self.data.iloc[indices]
        docs = [self.docs[i] for i in data.index]
        data = data.reset_index(drop=True)
        return DocumentSet(docs, data)

    def difference(self, other: DocumentSet) -> DocumentSet:
        haystack = [d.id for d in other]
        indices = []

        for i, doc in enumerate(self.docs):
            needle = doc.id
            if not any(needle.matches(x) for x in haystack):
                indices.append(i)

        return self.select(indices)

    def intersect(self, other: DocumentSet) -> DocumentSet:
        haystack = [d.id for d in other]
        indices = []

        for i, doc in enumerate(self.docs):
            needle = doc.id
            if any(needle.matches(x) for x in haystack):
                indices.append(i)

        return self.select(indices)

    def concat(self, other: DocumentSet) -> DocumentSet:
        # TODO: We need a better way to deal to handle the case where
        # the columns are not equal in some way.
        data = pd.concat([self.data, other.data])

        docs = self.docs + other.docs
        return DocumentSet(docs, data)

    def union(self, other: DocumentSet) -> DocumentSet:
        n = len(self)
        indices = list(range(n))
        haystack = [d.id for d in self]

        for i, doc in enumerate(other.docs):
            needle = doc.id

            if not any(needle.matches(x) for x in haystack):
                indices.append(n + i)

        return self.concat(other).select(indices)

    def unique(self) -> DocumentSet:
        indices = []

        for i, doc in enumerate(self.docs):
            found = False
            needle = doc.id

            for other in self:
                if other.id.matches(needle):
                    other._identifier = other._identifier.merge(needle)
                    found = True
                    break

            if not found:
                indices.append(i)

        return self.select(indices)

    def sample(self, n, seed=0) -> DocumentSet:
        if len(self) <= n:
            return self

        indices = random.sample(len(self), n)
        indices.sort()
        return self.select(indices)

    def itertuples(self):
        return zip(self.docs, self.data.itertuples())

    def __or__(self, other):
        return self.union(other)

    def __and__(self, other):
        return self.intersect(other)

    def __add__(self, other):
        return self.concat(other)

    def __sub__(self, other):
        return self.difference(other)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[key]
        elif isinstance(key, int):
            return self.docs[key]
        else:
            return self.select(key)

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
    def pubmed(self):
        return self._attr.get('pubmed')

    @property
    def arxivid(self):
        return self._attr.get('arxivid')

    @property
    def scopusid(self):
        return self._attr.get('eid')

    @property
    def s2id(self):
        return self._attr.get('s2id')

    def matches(self, other):
        n = 0

        # Two identifiers match if all keys that they have in common are equal
        for key in self._attr:
            if key in other._attr:
                if self._attr[key] != other._attr[key]:
                    return False
                n += 1

        if n > 0:
            return True

        return fuzzy_match(self._title, other._title)

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

    @property
    def citations(self) -> Optional[List[DocumentIdentifier]]:
        return None

    def mentions(self, term):
        pattern = r'(^|\s)' + re.escape(term) + r'($|\s)'
        flags = re.IGNORECASE
        keywords = self.keywords or []

        for text in [self.title, self.abstract] + keywords:
            if text and re.find(pattern, text, flags=flags):
                return True

        return False


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
    def s2id(self):
        return None

    @property
    def affiliations(self) -> 'Optional[list[Affiliation]]':
        return None


class DocumentMapping:
    def __init__(self, docs=None):
        self.title = dict()
        self.doi = dict()
        self.eid = dict()

    def add(self, doc, value):
        if doc.scopusid:
            self.eid[doc.scopusid] = value

        if doc.doi:
            self.doi[doc.doi] = value

        if doc.title:
            self.title[canonical(doc.title)] = value

    def get(self, doc):
        result = None

        if result is None and doc.scopusid:
            result = self.eid.get(doc.scopusid)

        if result is None and doc.doi:
            result = self.doi.get(doc.doi)

        if result is None and doc.title:
            result = self.title.get(canonical(doc.title))

        return result
