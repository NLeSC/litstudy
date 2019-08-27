from .common import Document, DocumentSet, Author

def search_mockup():
    a = Document(
            title=' A unified analytical theory of heteropolymers for sequence-specific phase behaviors of polyelectrolytes and polyampholytes ',
            authors=[
                Author(name='Yi-Hsuan Lin'),
                Author(name='Jacob P. Brady'),
                Author(name='Hue Sun Chan'),
                Author(name='Kingshuk Ghosh'),
            ],
            doi='arXiv:1908.09726',
            year=2019)

    b = Document(
            title='Scaling methods for accelerating kinetic Monte Carlo simulations of chemical reaction networks',
            authors=[
                Author(name='Yen Ting Lin'),
                Author(name='Song Feng'),
                Author(name='William S. Hlavacek'),
            ],
            year=2019)

    c = Document(
            title='Discreteness of populations enervates biodiversity in evolution',
            authors=[
                Author(name='Yen-Chih Lin'),
                Author(name='Tzay-Ming Hong'),
                Author(name='Hsiu-Hau Lin'),
            ],
            doi='arXiv:1005.4335',
            year=2010)

    d = Document(
            title='Phylogenetic Analysis of Cell Types using Histone Modifications',
            authors=[
                Author(name='Nishanth Ulhas Nair'),
                Author(name='Yu Lin'),
                Author(name='Philipp Bucher'),
                Author(name='Bernard M. E. Moret'),
            ],
            doi='arXiv:1307.7919',
            year=2013)


    return DocumentSet([a, b, c, d])

def search_scopus(query):
    return []
