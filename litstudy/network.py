import networkx as nx
import pyvis
import textwrap
import math
from .clean import canonical
from collections import defaultdict


class DocumentMapping:
    def __init__(self, docs=None):
        self.title = dict()
        self.doi = dict()
        self.eid = dict()

    def add(self, doc, value):
        if doc.eid:
            self.eid[doc.eid] = value

        if doc.doi:
            self.doi[doc.doi] = value

        if doc.title:
            self.title[canonical(doc.title)] = value


    def get(self, doc):
        result = None

        if result is None and doc.eid:
            result = self.eid.get(doc.eid)

        if result is None and doc.doi:
            result = self.doi.get(doc.doi)

        if result is None and doc.title:
            result = self.title.get(canonical(doc.title))

        return result


def build_citation_network(docs):
    g = nx.DiGraph()
    mapping = DocumentMapping()

    for i, doc in enumerate(docs):
        label = textwrap.fill(doc.title, width=20)
        g.add_node(i, label=label)
        mapping.add(doc.id, i)

    for i, doc in enumerate(docs):
        for ref in doc.references or []:
            j = mapping.get(ref)

            if j is not None:
                g.add_edge(i, j)

    return g


def build_cocitation_network(docs, max_edges=1000):
    g = nx.Graph()
    mapping = DocumentMapping()
    strength = defaultdict(int)

    for i, doc in enumerate(docs):
        label = textwrap.fill(doc.title, width=20)
        g.add_node(i, label=label)
        mapping.add(doc.id, i)

    for doc in docs:
        refs = []

        for ref in doc.references or []:
            j = mapping.get(ref)

            if j is not None:
                refs.append(j)

        for i in refs:
            for j in refs:
                if i < j:
                    strength[i,j] += 1

    if len(strength) > max_edges:
        strength = list(strength.items())
        strength.sort(key=lambda p: p[1], reverse=True)
        strength = strength[:max_edges]
        strength = dict(strength)

    for (i, j), weight in strength.items():
        g.add_edge(i, j, weight=weight)

    return g


def plot_network(g, **kwargs):
    g.remove_nodes_from(list(nx.isolates(g)))

    if len(g.edges) == 0:
        print('no edges given')
        return

    for n in g.nodes():
        g.nodes[n]['size'] = math.sqrt(g.degree(n) + 1)

    v = pyvis.network.Network(notebook=True, width='100%', height='750px')
    v.from_nx(g)
    v.show_buttons()
    return v.show('citation.html')

def plot_citation_network(docs, **kwargs):
    return plot_network(build_citation_network(docs), **kwargs)

