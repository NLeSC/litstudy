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
        return self.entry.get("Title")

    def _parse_authors(self, auths: str) -> List[str]:
        """
        helper method. parse two formats of Authors field.
        Formats:

            1. deliminated by , or ;

        if either deliminator isn't present, will assume
        only one author in field
        """
        if ";" in auths:
            author_list = auths.split("; ")
        elif "," in auths:
            author_list = auths.split(", ")
            # add comma between auth last, first to match format
            # in authors with affiliations
            author_list = [", ".join(auth.rsplit(" ", 1)) for auth in author_list]

        else:  # single author...
            author_list = [auths]
        return author_list

    def _get_authors_ids(self) -> List[str]:
        """
        helper method to parse two formats of
            'Author(s) ID' field

        1. AUTHOR_ID;AUTHOR_ID_2;LAST_AUTHOR_ID;
        2. AUTHOR_ID; AUTHOR_ID_2; LAST AUTHOR_ID
        """
        auths_id = self.entry.get("Author(s) ID")

        if auths_id == "":
            return []

        if auths_id[-1] == ";":
            auths_id = auths_id[:-1]

        auths_ids = auths_id.split(";")
        auths_ids = [auth_id.lstrip().rstrip() for auth_id in auths_ids]
        return auths_ids

    def _try_to_add_ids_to_authors(self, auths: List[str]) -> List[str]:
        auths_ids = self._get_authors_ids()

        if len(auths_ids) == len(auths) and len(auths) > 0:
            auths_w_ids = [f"{name} (ID: {auth_id})" for name, auth_id in zip(auths, auths_ids)]
            return auths_w_ids
        return auths

    def _parse_affiliations(self, affs) -> List[str]:
        if affs == "":
            return []
        return [aff.lstrip().rstrip() for aff in affs.split(";")]

    @property
    def authors(self) -> Optional[List[ScopusCsvAuthor]]:
        auths = self.entry.get("Authors")
        no_authors_formats = ["[No Authors Found]", "[No author name available]"]

        if auths == "" or auths in no_authors_formats:
            return None

        # use auths to search in auths_with_affs string.
        auths = self._parse_authors(auths)
        # use auths_with_ids for unique field.
        authors_with_ids = self._try_to_add_ids_to_authors(auths)

        affs = self.entry.get("Affiliations")
        affs = self._parse_affiliations(affs)
        # if single author, no way to know if ',' in author name
        # within auths_affs field (can't search string),
        # use 'Affiliations' field.
        if len(auths) == 1:
            return [
                ScopusCsvAuthor(authors_with_ids[0], [ScopusCsvAffiliation(aff) for aff in affs])
            ]

        auths_affs = self.entry.get("Authors with affiliations")
        if auths_affs == "":  # can't map affiliations to authors
            return [ScopusCsvAuthor(auth, None) for auth in authors_with_ids]

        indexes_of_authors = [auths_affs.index(auth) for auth in auths]
        auth_to_affs_mapping = {}

        for num, index in enumerate(indexes_of_authors):
            # auth = auths[num]

            if num < len(indexes_of_authors) - 1:
                next_index = indexes_of_authors[num + 1]
                cur_auth_affils = auths_affs[index:next_index]
            else:
                cur_auth_affils = auths_affs[index:]
            # cur_auth_affils = substring.replace(f"{auth}, ", "")
            # could be multiple affiliates, but no clear deliminator

            affs_filtered = [a for a in affs if a in cur_auth_affils]
            affs_filtered = sorted(affs_filtered, key=lambda x: len(x))
            # edge case is str in affs is substr of aff in cur_auth_affs

            # removes edge case where aff is substring of other aff
            disclude = []
            short_string = affs_filtered[0]
            for j in range(0, len(affs_filtered) - 1):
                long_strings = affs_filtered[j + 1 :]
                for ls in long_strings:
                    if short_string in ls:
                        disclude.append(short_string)
                short_string = affs_filtered[j + 1]

            auth_to_affs_mapping[authors_with_ids[num]] = [
                ScopusCsvAffiliation(a) for a in affs_filtered if a not in disclude
            ]
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
            return []
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
