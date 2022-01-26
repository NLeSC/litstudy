Data Types
------------------------------------------

There are two core datatypes in litstudy: `Document` and `DocumentSet`.

`Document` is an abstract base class (ABC) that provides access to the metadata of documents in a unified way.
Different backends provide their own implements of this class (for example, `ScopusDocument`, `BibTexDocument`, etc.)

`DocumentSet` is set of `Document` objects.
All set operations are supported, making it possible to create a new set from existing sets.
For instance, it is possible to load documents from two sources (obtaining two `DocumentSets`) and merge them (obtaining one large `DocumentSet`).


.. automodule:: litstudy.types
  :members:

