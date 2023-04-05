import hashlib
import json
import os
import pickle
import requests
from litstudy.types import Document, DocumentIdentifier, DocumentSet


class MockResponse:
    def __init__(self, data):
        self.data = data

    @property
    def status_code(self):
        return self.data["status_code"]

    @property
    def content(self):
        return self.data["content"]

    def json(self):
        return json.loads(self.content)


class MockSession:
    def __init__(self, directory=None, allow_requests=None):
        if directory is None:
            directory = os.path.dirname(os.path.realpath(__file__)) + "/requests/"

        if allow_requests is None:
            allow_requests = bool(os.environ.get("LITSTUDY_ALLOW_REQUESTS", False))

        self.directory = directory
        self.allow_requests = allow_requests

    def _clean_url(self, url):
        return hashlib.sha1(url.encode("utf8")).hexdigest()

    def get(self, url):
        filename = os.path.join(self.directory, self._clean_url(url) + ".pickle")

        if not os.path.exists(filename):
            if self.allow_requests:
                response = requests.get(url)
                data = dict(
                    url=url,
                    status_code=response.status_code,
                    content=response.content,
                )

                with open(filename, "wb") as f:
                    f.write(pickle.dumps(data))
            else:
                raise KeyError(f"URL not registered with MockSession: {url}")

        with open(filename, "rb") as f:
            return MockResponse(pickle.loads(f.read()))


class ExampleDocument(Document):
    def __init__(self, id: DocumentIdentifier):
        super().__init__(id)

    @property
    def title(self):
        return self.id.title

    @property
    def authors(self):
        return None


def example_docs() -> DocumentSet:
    a = DocumentIdentifier(
        "The European Approach to the Exascale Challenge", doi="10.1109/MCSE.2018.2884139"
    )

    b = DocumentIdentifier(
        "this document should not exists since the title is long",
    )

    return DocumentSet([ExampleDocument(a), ExampleDocument(b)])
