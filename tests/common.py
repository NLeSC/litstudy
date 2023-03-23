import json
import sys
import re
import os
import requests
import pickle
import hashlib


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
