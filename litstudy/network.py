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
