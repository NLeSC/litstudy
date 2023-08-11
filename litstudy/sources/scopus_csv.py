"""
support loading Scopus CSV export.
"""
from typing import List, Optional
from ..types import Document, Author, DocumentSet, DocumentIdentifier, Affiliation
from ..common import robust_open
import csv


class ScopusCsvAffiliation(Affiliation):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name


class ScopusCsvAuthor(Author):
    def __init__(self, name, affiliations):
        self._name = name
        self._affiliations = affiliations

    @property
    def name(self):
        return self._name

    @property
    def affiliations(self):
        return self._affiliations


class ScopusCsvDocument(Document):
    def __init__(self, entry):
        doi = entry.get("DOI")
        title = entry.get("Title")
        pubmed_id = entry.get("PubMed ID")
        eid = entry.get("EID")
        identifier = DocumentIdentifier(title, doi=doi, pubmed=pubmed_id, eid=eid)
        super().__init__(identifier)
        self.entry = entry

    @property
    def title(self) -> Optional[str]:
        return self.entry.get("Title") or None

    def _get_authors(self) -> Optional[List[str]]:
        """
        helper method. parse two formats of Authors field.
        Formats:
            1. deliminates author by , and [No author name available]
            2. deliminates author by ; and [No Authors found]
        """
        auths = self.entry.get("Authors")
        no_authors_formats = ["[No Authors Found]", "[No author name available]"]

        if auths in no_authors_formats:
            return []

        if ";" in auths:
            author_list = auths.split("; ")
        elif "," in auths:
            author_list = auths.split(", ")
            # add comma between auth last, first to match format
            # in authors with affiliations
            author_list = [', '.join(auth.rsplit(" ", 1)) for auth in author_list]
        else: # single author...
            author_list = [auths]

        return author_list

    def _get_authors_ids(self) -> Optional[List[str]]:
        """
        helper method to parse two formats of
            'Author(s) ID' field
        1. AUTHOR_ID;AUTHOR_ID_2;LAST_AUTHOR_ID;
        2. AUTHOR_ID; AUTHOR_ID_2; LAST AUTHOR_ID
        """
        auths_id = self.entry.get("Author(s) ID", "")
        if auths_id[-1] == ";":
            auths_id = auths_id[:-1]
        auths_ids = auths_id.split(";")
        auths_ids = [auth_id.lstrip().rstrip() for auth_id in auths_ids]
        return auths_ids

    def _get_auths_with_ids(self):
        auths = self._get_authors()
        auths_id = self._get_authors_ids()

        if len(auths) == len(auths_id):
            auths_w_ids = [f"{name} (ID: {auth_id})" for name, auth_id in zip(auths, auths_id)]
        else:
            auths_w_ids = []
        return auths_w_ids

    @staticmethod
    def _get_affiliations(affiliation_substring):
        affiliations = affiliation_substring.split(";")
        affiliations = [
            ScopusCsvAffiliation(aff) for aff in affiliations]
        return affiliations

    @property
    def authors(self) -> List[ScopusCsvAuthor]:
        auths = self._get_authors()
        auths_affs = self.entry.get("Authors with affiliations")
        auths_ids = self._get_authors_ids()
        affs = self.entry.get("Affiliations")
        # author_last, first initial, affiliation; .....
        # try to add id to author name
        if not auths_affs or auths is None or auths_ids is None:
            return []

        auths_w_ids = self._get_auths_with_ids()
        # if single author, no way to know if ',' in author name
        # within auths_affs field,
        # so need to use 'Affiliations' field instead
        # of searching for author in auths_affs
        if len(auths) == 1:
            return [ScopusCsvAuthor(auths[0], self._get_affiliations(affs))]
        print(auths)
        print(auths_affs)
        indexes_of_authors = [auths_affs.index(auth) for auth in auths]
        auth_to_affs_mapping = {}
        for num, index in enumerate(indexes_of_authors):
            auth = auths[num]
            if num < len(indexes_of_authors) - 1:

                next_index = indexes_of_authors[num+1]
                substring = auths_affs[index:next_index]
                # only want part of string for current auhtor
                # and affiliations
            else:
                substring = auths_affs[index:]

            substring = substring.replace(f"{auth}, ", "")
            affiliations = substring.split(";")
            affiliations = [
                ScopusCsvAffiliation(aff) for aff in affiliations]
            if auths_w_ids!=[]:
                auth_to_affs_mapping[auths_w_ids[num]] = affiliations
            else:
                auth_to_affs_mapping[auth] = affiliations

        return [ScopusCsvAuthor(a, b) for a, b in auth_to_affs_mapping.items()]

    @property
    def publisher(self) -> Optional[str]:
        return self.entry.get("Publisher") or None

    @property
    def publication_year(self) -> Optional[int]:
        year = self.entry.get("Year")
        if not year:
            return None

        try:
            return int(year)
        except:
            return None

    @property
    def keywords(self) -> Optional[List[str]]:
        keywords = self.entry.get("Author Keywords")
        if not keywords:
            return None
        return keywords.split("; ")

    @property
    def abstract(self) -> Optional[str]:
        abstract = self.entry.get("Abstract")
        if not abstract:
            return None
        return abstract

    @property
    def citation_count(self) -> Optional[int]:
        citation_count = self.entry.get("Cited by")
        if not citation_count:
            return None
        return int(citation_count)

    @property
    def language(self) -> Optional[str]:
        return self.entry.get("Language of Original Document") or None

    @property
    def publication_source(self) -> Optional[str]:
        return self.entry.get("Source title") or None

    @property
    def source_type(self) -> Optional[str]:
        return self.entry.get("Document Type") or None


def load_scopus_csv(path: str) -> DocumentSet:
    """Import CSV file exported from Scopus"""
    with robust_open(path) as f:
        lines = csv.DictReader(f)
        docs = [ScopusCsvDocument(line) for line in lines]
        return DocumentSet(docs)
