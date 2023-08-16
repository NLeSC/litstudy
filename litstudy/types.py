from abc import ABC, abstractmethod
from datetime import date
from typing import Optional, List
import numpy as np
import pandas as pd
import random
import re

from .common import fuzzy_match, canonical, progress_bar


class DocumentSet:
    """Represents a set of documents.

    `DocumentSet` stores a list of `Document` objects. Optionally, a pandas
    data frame can be provided which stores additional properties on the
    documents.

    All set operations are accepted by `DocumentSet` (union, intersection,
    difference), allowing for new sets to be created from existing sets.

    Note that a `DocumentSet` is immutable and its content cannot be changed.
    Instead, most methods below return a new `DocumentSet` instead of
    performing modifications in-place.
    """

    def __init__(self, docs, data=None):
        """Construct a new `DocumentSet`.

        :param docs: A list (or iterator) of `Document` objects.
        :param data: Additional metadata associated with the documents. This
                     can be either a `pandas.DataFrame` or something which
                     is accepted by the pandas DataFrame constructor.
        """
        docs = list(docs)

        if data is None:
            data = pd.DataFrame(index=range(len(docs)))
        else:
            data = pd.DataFrame(data)

        assert len(data) == len(docs)
        self.data = data
        self.docs = docs

    def _refine_docs(self, callback):
        new_indices = []
        new_docs = []
        old_indices = []
        old_docs = []

        for i, doc in enumerate(progress_bar(self.docs)):
            new_doc = callback(doc)

            if new_doc is not None:
                new_indices.append(i)
                new_docs.append(new_doc)
            else:
                old_indices.append(i)
                old_docs.append(doc)

        new_data = self.data.iloc[new_indices]
        old_data = self.data.iloc[old_indices]

        # FIX: not forget to reset the index
        new_data.reset_index(drop=True, inplace=True)
        old_data.reset_index(drop=True, inplace=True)

        return DocumentSet(new_docs, new_data), DocumentSet(old_docs, old_data)

    def add_property(self, name: str, values) -> "DocumentSet":
        """Returns a new set which has an additional property added.

        :param name: Name of the new property.
        :param values: List of values. Should be the same length as the
                       number of documents in this set.
        :returns: The new document set.
        """
        data = self.data.copy(deep=False)
        data[name] = values
        return DocumentSet(self.docs, data)

    def remove_property(self, name: str) -> "DocumentSet":
        """Returns a new set which has the given property removed.

        :param name: Name of the property.
        :returns: The new document set.
        """
        data = self.data.copy(deep=False)
        data.drop(name)
        return DocumentSet(self.docs, data)

    def filter_docs(self, predicate) -> "DocumentSet":
        """Returns a new set for which the provided predicate returned `True`.

        :param predicate: A function `Document -> bool`.
        :returns: The new document set.
        """
        return self.filter(lambda doc, _: predicate(doc))

    def filter(self, predicate) -> "DocumentSet":
        """Returns a new set for which the provided predicate returned `True`.

        :param predicate: A function `Document, dict -> bool`. The provided
                          dict stores the properties of the document.
        :returns: The new document set.
        """
        indices = []

        for doc, record in zip(self.docs, self.data.itertuples()):
            if predicate(doc, record):
                indices.append(record.Index)

        return self.select(indices)

    def select(self, indices) -> "DocumentSet":
        """Returns a new set which contains only the documents at the
        provided indices.

        :param indices: Any input accepted by `pandas.DataFrame.iloc` such
                        as a list of integer.
        :returns: The new document set.
        """

        data = self.data.iloc[indices]
        docs = [self.docs[i] for i in data.index]
        data = data.reset_index(drop=True)
        return DocumentSet(docs, data)

    def _intersect_indices(self, other):
        haystack = [d.id for d in other]
        left = []
        right = []

        for i, doc in enumerate(self):
            needle = doc.id

            for j, id in enumerate(haystack):
                if id.matches(needle):
                    left.append(i)
                    right.append(j)
                    break

        return left, right

    def _zip_with(self, left, other, right):
        assert len(left) == len(right)
        data = dict()

        for key, column in self.data.iloc[left].items():
            column = column.copy().reset_index(drop=True)
            data[key] = column

        for key, column in other.data.iloc[right].items():
            column = column.copy().reset_index(drop=True)

            if key in data:
                data[key] = np.maximum(data[key], column)
            else:
                data[key] = column

        return pd.DataFrame(index=range(len(left)), data=data)

    def intersect(self, other: "DocumentSet") -> "DocumentSet":
        """Returns a new set which contains the documents provided in
        both `self` and `other`. This is also available as the `&` operator.

        :returns: The new document set.
        """
        if not self or not other:
            return DocumentSet([])

        left, right = DocumentSet._intersect_indices(self, other)

        docs = [self.docs[i] for i in left]  # Select docs from left?
        data = DocumentSet._zip_with(self, left, other, right)
        return DocumentSet(docs, data)

    def difference(self, other: "DocumentSet") -> "DocumentSet":
        """Returns a new set which contains the documents provided in
        `self` but not in `other`. This is also available as the `-` operator.

        :returns: The new document set.
        """
        if not other or not self:
            return self

        indices, _ = DocumentSet._intersect_indices(self, other)
        return self.select(sorted(set(range(len(self))) - set(indices)))

    def union(self, other: "DocumentSet") -> "DocumentSet":
        """Returns a new set which contains the documents provided in
        either `self` and `other`. Duplicate documents in `other` that also
        appear in `self` are discarded. This is also available as the `|`
        operator.

        :returns: The new document set.
        """
        if not other:
            return self
        if not self:
            return other

        left, right = DocumentSet._intersect_indices(self, other)
        if not left:
            return DocumentSet.concat(self, other)

        docs = [self.docs[i] for i in left]  # Select docs from left?
        data = DocumentSet._zip_with(self, left, other, right)
        middle = DocumentSet(docs, data)

        left = self.select(sorted(set(range(len(self))) - set(left)))
        right = other.select(sorted(set(range(len(other))) - set(right)))

        return DocumentSet.concat(middle, DocumentSet.concat(left, right))

    def concat(self, other: "DocumentSet") -> "DocumentSet":
        """Returns a new set which does contain the documents provided in
        either `self` and `other`. Duplicate documents are not removed, see
        `union` instead. This is also available as the `+` operator.

        :returns: The new document set.
        """

        def default_val(dtype):
            c = dtype.char
            if c == "?":
                return False
            if c in "bBiu":
                return 0
            if c in "fc":
                return float("nan")
            if c in "SaU":
                return ""

            return None

        if not self:
            return other

        if not other:
            return self

        left = self.data.copy()
        right = other.data.copy()

        for col in left:
            if col not in right:
                dtype = left[col].dtype
                right[col] = np.array(default_val(dtype)).astype(dtype)

        for col in right:
            if col not in left:
                dtype = right[col].dtype
                left[col] = np.array(default_val(dtype)).astype(dtype)

        data = pd.concat([left, right], ignore_index=True)
        docs = self.docs + other.docs
        return DocumentSet(docs, data)

    def unique(self) -> "DocumentSet":
        """Returns a new set which has all duplicate documents removed.

        :returns: The new document set.
        """
        indices = []

        for i, doc in enumerate(self.docs):
            found = False
            needle = doc.id

            for other in self[:i]:
                if other.id.matches(needle):
                    other._identifier = other._identifier.merge(needle)
                    found = True
                    break

            if not found:
                indices.append(i)

        return self.select(indices)

    def sample(self, n, seed=0) -> "DocumentSet":
        """Returns a new set which contains `n` randomly chosen documents
        from `self`.

        :returns: The new document set.
        """
        if len(self) <= n:
            return self

        random.seed(seed)
        indices = random.sample(len(self), n)
        indices.sort()
        return self.select(indices)

    def itertuples(self):
        """Returns an iterator over `(Document, dict)` tuples, where the
        `dict` contains the properties of this document.
        """
        return zip(self.docs, self.data.itertuples())

    def __or__(self, other):
        """Alias for `DocumentSet.union`"""
        return self.union(other)

    def __and__(self, other):
        """Alias for `DocumentSet.intersect`"""
        return self.intersect(other)

    def __add__(self, other):
        """Alias for `DocumentSet.concat`"""
        return self.concat(other)

    def __sub__(self, other):
        """Alias for `DocumentSet.difference`"""
        return self.difference(other)

    def __len__(self):
        """Returns the number of documents in this set"""
        return len(self.docs)

    def __getitem__(self, key):
        """Returns different things depending on the key type:

        * `str`: The property named `key` is returned.
        * `int`: The document at position `key` is returned.
        * otherwise: The call is forwarded to `DocumentSet.select`.
        """
        if isinstance(key, str):
            return self.data[key]
        elif np.issubdtype(type(key), np.integer):  # any type of integer works
            return self.docs[int(key)]
        else:
            return self.select(key)

    def __iter__(self):
        """Returns an iterator over `Document` objects in this set."""
        return iter(self.docs)

    def __bool__(self):
        return bool(len(self))

    def __repr__(self):
        return f"<{len(self)} documents>"


class DocumentIdentifier:
    """Represents an identifier for a document.

    Uniquely identifing an scientific document is often difficult since a
    single document might have multiple identifiers assigned to it (e.g., DOI,
    PubMed ID, Scopus ID, SemanticScholar ID) and not all data sources might
    provide all these identifiers. This class stores all possible identifiers
    that a document has.
    """

    def __init__(self, title, **attr):
        # Remove keys where value is None
        self._title = title
        self._attr = dict((k, v) for k, v in attr.items() if v)

    @property
    def title(self) -> Optional[str]:
        """Returns the title."""
        return self._title

    @property
    def doi(self) -> Optional[str]:
        """Returns the DOI (example: 10.1093/ajae/aaq063)."""
        return self._attr.get("doi")

    @property
    def pubmed(self) -> Optional[str]:
        """Returns the PubMed ID."""
        return self._attr.get("pubmed")

    @property
    def arxivid(self) -> Optional[str]:
        """Returns the arXiv ID."""
        return self._attr.get("arxivid")

    @property
    def scopusid(self) -> Optional[str]:
        """Returns the Scopus ID."""
        return self._attr.get("eid")

    @property
    def s2id(self) -> Optional[str]:
        """Returns the Semantic Scholar ID."""
        return self._attr.get("s2id")

    def matches(self, other: "DocumentIdentifier") -> bool:
        """Returns `True` iff these two identifiers are equivalent


        Two documents are considered to be equivalent if all identifiers they
        have in common are equal. For example, if both documents have a DOI
        then these should be the same. If two documents have not a single
        identifier in common, a fuzzy match based on the title is performed.
        """
        n = 0

        # Two identifiers match if all keys that they have in common are equal
        for key in self._attr:
            if key in other._attr:
                if self._attr[key] != other._attr[key]:
                    return False
                n += 1

        if n > 0:
            return True

        # No identifiers in common
        return fuzzy_match(self._title, other._title)

    def merge(self, other) -> "DocumentIdentifier":
        """Returns a new `DocumentIdentifier` which adds the identifiers
        `others` to `self`.
        """
        attr = dict()
        attr.update(other._attr)
        attr.update(self._attr)
        return DocumentIdentifier(self._title, **attr)

    def __repr__(self):
        return f"<{self._title}, {self._attr}>"


class Document(ABC):
    """Stores the metadata of a document.

    This is an interface which provides several methods which can be
    overridden by child classes. All methods can thus return `None`
    in case that method is not overridden.
    """

    def __init__(self, identifier: DocumentIdentifier):
        self._identifier = identifier

    @property
    def id(self) -> DocumentIdentifier:
        """The `DocumentIdentifier` of this document."""
        return self._identifier

    @property
    @abstractmethod
    def title(self) -> str:
        """The title of this document."""
        pass

    @property
    @abstractmethod
    def authors(self) -> Optional[List["Author"]]:
        """The authors of this document."""
        pass

    @property
    def affiliations(self) -> Optional[List["Affiliation"]]:
        """The affiliations associated with the authors of this document."""
        authors = self.authors

        if authors is None:
            return None

        items = dict()
        for author in authors:
            affiliations = author.affiliations

            if affiliations:
                for aff in affiliations:
                    if aff.name:
                        items[aff.name] = aff

        return list(items.values())

    @property
    def publisher(self) -> Optional[str]:
        """The publisher of this document."""
        return None

    @property
    def language(self) -> Optional[str]:
        """The language this document is written in."""
        return None

    @property
    def publication_date(self) -> Optional[date]:
        """The data of publication."""
        return None

    @property
    def publication_year(self) -> Optional[int]:
        """The year of publication."""
        date = self.publication_date
        if date is None:
            return None
        return date.year

    @property
    def publication_source(self) -> Optional[str]:
        """The name of the publication source (i.e., journal name,
        conference name, etc.)
        """
        return None

    @property
    def source_type(self) -> Optional[str]:
        """The type of publication source (i.e., journal, conference
        proceedings, book, etc.)
        """
        return None

    @property
    def keywords(self) -> Optional[List[str]]:
        """The keywords of this document. What exactly consistutes as
        keywords depends on the data source (author keywords, generated
        keywords, topic categories), but is should be a list of strings.
        """
        return None

    @property
    def abstract(self) -> Optional[str]:
        """The abstract of this document."""
        return None

    @property
    def citation_count(self) -> Optional[int]:
        """The number of citations that this document received."""
        return None

    @property
    def references(self) -> Optional[List[DocumentIdentifier]]:
        """The list of other documents that are cited by this document."""
        return None

    @property
    def citations(self) -> Optional[List[DocumentIdentifier]]:
        """The list of other documents that cite this document."""
        return None

    def mentions(self, term: str) -> bool:
        """Returns `True` if this document mentions the given term in the
        title, abstract, or keywords.
        """
        pattern = r"(^|\s)" + re.escape(term) + r"($|\s)"
        flags = re.IGNORECASE
        keywords = self.keywords or []

        for text in [self.title, self.abstract] + keywords:
            if text and re.search(pattern, text, flags=flags):
                return True

        return False


class Affiliation(ABC):
    """Represents the affiliation of an author"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the affiliation"""
        pass

    @property
    def city(self) -> Optional[str]:
        """City the affiliation is located in."""
        pass

    @property
    def country(self) -> Optional[str]:
        """Country the affiliation is located in."""
        pass


class Author(ABC):
    """Represents the author of a document."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the author."""
        pass

    @property
    def orcid(self) -> Optional[str]:
        """The ORCID of the author."""
        return None

    @property
    def s2id(self) -> Optional[str]:
        """The SemanticScholar ID of the author."""
        return None

    @property
    def affiliations(self) -> "Optional[list[Affiliation]]":
        """The affiliations this author is associated with."""
        return None


class DocumentMapping:
    def __init__(self, docs=None):
        self.title = dict()
        self.doi = dict()
        self.eid = dict()

        if docs:
            for index, doc in enumerate(docs):
                self.add(doc.id, index)

    def add(self, doc: DocumentIdentifier, value):
        if doc.scopusid:
            self.eid[doc.scopusid] = value

        if doc.doi:
            self.doi[doc.doi] = value

        if doc.title:
            self.title[canonical(doc.title)] = value

    def get(self, doc: DocumentIdentifier):
        result = None

        if result is None and doc.scopusid:
            result = self.eid.get(doc.scopusid)

        if result is None and doc.doi:
            result = self.doi.get(doc.doi)

        if result is None and doc.title:
            result = self.title.get(canonical(doc.title))

        return result
