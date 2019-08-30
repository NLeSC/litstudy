import networkx


def build_citation_network(docs):
    title2index = dict()
    g = networkx.DiGraph()

    for i, doc in enumerate(docs):
        g.add_node(i, document=doc, label=doc.title)
        title2index[doc.title] = i

    for i, doc in enumerate(docs):
        if doc.references:
            for ref in doc.references:
                if ref in title2index:
                    g.add_edge(title2index[ref], i)

    return g
                

def plot_citation_network(docs, **kwargs):
    """ Plot the citation network of the given `DocumentSet` where nodes
    represent document and edges represents citations between documents.

    :param docs: The `DocumentSet`
    :param \**kwargs: Additional arguments passed to `networkx.draw`
    """
    g = build_citation_network(docs)

    if len(g.edges) == 0:
        print('Citations not available for given document set')
        return

    options = dict(
    )
    options.update(kwargs)
    networkx.draw(g, **options)


def build_coauthor_network(docs):
    g = networkx.Graph()
    n = 0
    nodes = dict()
    weights = []
    edges = dict()

    for doc in docs:
        authors = set()

        if doc.authors:
            for author in doc.authors:
                authors.add(author.name)

        for author in authors:
            if author not in nodes:
                nodes[author] = n
                weights.append(1)
                n += 1
            else:
                weights[nodes[author]] += 1

        for a in authors:
            for b in authors:
                key = (nodes[a], nodes[b])
                if key[0] < key[1]:
                    if key not in edges:
                        edges[key] = 1
                    else:
                        edges[key] += 1

    g.add_nodes_from((i, dict(author=a, weight=weights[i])) for (a, i) in nodes.items())
    g.add_edges_from((i, j, dict(weight=w)) for ((i, j), w) in edges.items())

    return g

def plot_coauthor_network(docs, top_k=25, min_degree=1, **kwargs):
    """ Plot the citation network of the given `DocumentSet` where nodes
    represent author and edges represents collaborations (e.g. number of
    shared documents between two authors.)

    :param docs: The `DocumentSet`
    :param min_degree: Only show nodes of at least this degree, defaults to 1
    :param top_k: Only show labels for the top k nodes.
    :param \**kwargs: Additional arguments passed to `networkx.draw`
    """
    g = build_coauthor_network(docs)

    deg = dict(g.degree())
    valid = [k for k in deg if deg[k] >= min_degree]
    max_deg = float(max(deg.values()))

    top_authors = sorted(deg, key=lambda k: deg[k], reverse=True)[:top_k]
    labels = dict((n, g.nodes[n]['author']) for n in top_authors if n in valid)

    options = dict(
            nodelist=valid,
            node_size=[deg[k] + 1 for k in valid],
            labels=labels,
            edge_color='darkgray',
    )
    options.update(kwargs)
    networkx.draw(g, **options)
    
