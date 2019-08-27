from .common import Document, DocumentSet, Author, Affiliation

def search_mockup():
    a = Document(
            title=' A unified analytical theory of heteropolymers for sequence-specific phase behaviors of polyelectrolytes and polyampholytes ',
            authors=[
                Author(name='Yi-Hsuan Lin', affiliations=[Affiliation(name='University of affiliation1'), Affiliation(name='affiliation3')]),
                Author(name='Jacob P. Brady', affiliations=[Affiliation(name='University of affiliation1')]),
                Author(name='Hue Sun Chan', affiliations=[Affiliation(name='affiliation2')]),
                Author(name='Kingshuk Ghosh'),
            ],
            id='arXiv:1908.09726',
            year=2019,
            source_type='type1',
            source='International Conference on 1')

    b = Document(
            title='Scaling methods for accelerating kinetic Monte Carlo simulations of chemical reaction networks',
            authors=[
                Author(name='Yen Ting Lin', affiliations=[Affiliation(name='affiliation2')]),
                Author(name='Song Feng', affiliations=[Affiliation(name='affiliation3')]),
                Author(name='William S. Hlavacek'),
            ],
            id='1',
            year=2019,
            source_type='type2',
            source='International Conference on 2')

    c = Document(
            title='Discreteness of populations enervates biodiversity in evolution',
            authors=[
                Author(name='Yen-Chih Lin'),
                Author(name='Tzay-Ming Hong'),
                Author(name='Hsiu-Hau Lin'),
            ],
            id='arXiv:1005.4335',
            year=2010,
            source_type='type2',
            source='International Conference on 2')

    d = Document(
            title='Phylogenetic Analysis of Cell Types using Histone Modifications',
            authors=[
                Author(name='Nishanth Ulhas Nair'),
                Author(name='Yu Lin'),
                Author(name='Philipp Bucher'),
                Author(name='Bernard M. E. Moret'),
            ],
            id='arXiv:1307.7919',
            year=2013,
            source_type='type3',
            source='International Conference on 3')


    return DocumentSet([a, b, c, d])

def search_scopus(query):
    return []
