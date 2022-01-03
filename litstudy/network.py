from .common import canonical
from .sources.types import DocumentMapping
from collections import defaultdict
import json
import math
import numpy as np
import networkx as nx
import pyvis
import textwrap


def plot_network(g, height='1000px', smooth=None, max_node_size=100,
                 min_node_size=5, **kwargs):
    g.remove_nodes_from(list(nx.isolates(g)))

    if len(g.edges) == 0:
        print('no edges given')
        return

    directed = nx.is_directed(g)
    v = pyvis.network.Network(
            notebook=True,
            width='100%',
            height=height,
            directed=directed
    )

    sizes = [attr.get('weight') for (_, attr) in g.nodes.items()]

    if all(s is not None for s in sizes):
        sizes = np.array(sizes)
    elif directed:
        sizes = [g for (_, g) in g.in_degree]
    else:
        sizes = [g for (_, g) in g.degree]

    sizes = np.array(sizes, dtype=np.float32)
    # sizes = np.sqrt(sizes)
    ratio = (max_node_size - min_node_size) / np.amax(sizes)
    sizes = ratio * sizes + min_node_size

    for id, size in zip(g, sizes):
        attr = g.nodes[id]
        v.add_node(
                id,
                title=attr['title'],
                label=textwrap.fill(attr['title'], width=20),
                shape='dot',
                size=float(size),
                color=attr.get('color'),
                labelHighlightBold=True,
        )

    edges = []
    weights = []

    for src, dst in g.edges():
        weight = g[src][dst].get('weight')
        if weight is not None:
            width = weight
            label = str(weight)
        else:
            width = None
            label = ''

        v.add_edge(src, dst, width=width, title=label)

    if smooth is None:
        smooth = len(g.edges()) < 1000

    v.set_options(json.dumps({
        'configure': {
            'enabled': True,
        },
        'nodes': {
            'font': {
                'size': 7,
            }
        },
        'edges': {
            'smooth': smooth,
            'color': {
                'opacity': 0.25,
            }
        },
        'physics': {
            'forceAtlas2Based': {
                'springLength': 100,
            },
            'solver': 'forceAtlas2Based',
        }
    }))

    return v.show('citation.html')


def build_base_network(docs, directed):
    g = nx.DiGraph() if directed else nx.Graph()
    mapping = DocumentMapping()

    for i, doc in enumerate(docs):
        g.add_node(i, title=doc.title)
        mapping.add(doc.id, i)

    return g, mapping


def build_citation_network(docs):
    g, mapping = build_base_network(docs, True)

    for i, doc in enumerate(docs):
        for ref in doc.references or []:
            j = mapping.get(ref)

            if j is not None:
                g.add_edge(i, j)

    return g


def plot_citation_network(docs, **kwargs):
    return plot_network(build_citation_network(docs), **kwargs)


def build_cocitation_network(docs, max_edges=None):
    max_edges = max_edges or 1000

    g, mapping = build_base_network(docs, False)
    strength = defaultdict(int)

    for doc in docs:
        refs = []

        for ref in doc.references or []:
            j = mapping.get(ref)

            if j is None:
                mapping.add(ref)
                j = mapping.get(ref)

            if j is not None:
                refs.append(j)

        for i in refs:
            for j in refs:
                if i < j:
                    strength[i, j] += 1

    edges = list(strength.items())

    if len(edges) > max_edges:
        edges.sort(key=lambda p: p[1], reverse=True)
        edges = edges[:max_edges]

    for (i, j), weight in edges:
        g.add_edge(i, j, weight=weight)

    return g


def plot_cocitation_network(docs, max_edges=None, node_size=10, **kwargs):
    return plot_network(
            build_cocitation_network(docs, max_edges),
            min_node_size=node_size,
            max_node_size=node_size,
            **kwargs
    )


def build_coupling_network(docs, max_edges=1000):
    g, mapping = build_base_network(docs, False)
    n = len(g)
    doc_refs = []

    for doc in docs:
        refs = []

        for ref in doc.references or []:
            i = mapping.get(ref)

            if i is None:
                mapping.add(ref, n)
                n += 1
                i = n

            if i is not None:
                refs.append(i)

        doc_refs.append(set(refs))

    strength = defaultdict(int)

    for i, a in enumerate(doc_refs):
        for j, b in enumerate(doc_refs[:i]):
            common = a & b

            if common:
                strength[i, j] = len(common)

    edges = list(strength.items())

    if len(edges) > max_edges:
        edges.sort(key=lambda p: p[1], reverse=True)
        edges = edges[:max_edges]

    for (i, j), weight in edges:
        g.add_edge(i, j, weight=weight, score=weight)

    return g


def plot_coupling_network(docs, max_edges=None, node_size=10, **kwargs):
    return plot_network(
            build_coupling_network(docs, max_edges),
            min_node_size=node_size,
            max_node_size=node_size,
            **kwargs
    )


def build_coauthor_network(docs, max_authors=1000):
    g = nx.DiGraph()
    paper_authors = []
    count = defaultdict(int)

    for doc in docs:
        authors = []

        for author in doc.authors or []:
            name = author.name

            if name:
                count[name] += 1

    authors = list(count.keys())

    if len(authors) > max_authors:
        authors.sort(key=lambda name: count[name], reverse=True)
        authors = authors[:max_authors]

    mapping = dict()
    for i, author in enumerate(authors):
        g.add_node(i, title=author)
        mapping[author] = i

    edges = defaultdict(int)

    for doc in docs:
        authors = [a.name for a in doc.authors or [] if a.name]

        for i, left in enumerate(authors):
            for right in authors[:i]:
                if left in mapping and right in mapping:
                    edges[mapping[left], mapping[right]] += 1

    for (left, right), weight in edges.items():
        g.add_edge(left, right, weight=weight)

    return g


def plot_coauthor_network(docs, max_authors=1000):
    return plot_network(
            build_coauthor_network(docs, max_authors),
    )
