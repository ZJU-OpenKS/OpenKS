import bisect
import json
import logging
import os

from nltk.corpus import stopwords as stop_words
from nltk.stem import SnowballStemmer

from features import Indexer

stopwords = stop_words.words('english')
stem = SnowballStemmer('english')
paper_threshold = 5

class Paper:
    def __init__(self, year):
        self.id = None
        self.authors = []
        self.year = year
        self.venue = None
        self.references = []
        self.terms = []

def parse_term(title):
    title = title.replace('_', ' ')
    title = title.replace(':', ' ')
    title = title.replace(';', ' ')
    wlist = title.strip().split()
    token = [j for j in wlist if j not in stopwords]
    token = stem.stemWords(token)
    return token

def genereate_papers(datafile, feature_begin, feature_end, observation_begin, observation_end, conf_list):
    indexer = Indexer(['authors', 'paper', 'term', 'venue'])

    index, authors, title, year, venue = None, None, None, None, None
    references = []

    n_authors = 0
    n_citation = 0
    n_terms = 0
    published = 0

    min_year = 3000
    max_year = 0

    papers_feature_window = []
    papers_observation_window = []

    with open(datafile, 'r') as file:
        datalist = json.load(file)
        for data in datalist:
            id = data['id']
            title = data['title']
            authors = data['authors']
            venue = data['venue']
            year = data['year']
            n_citation = data['n_citation']
            references = data['references']
            abstract = data['abstract']

            if year > 0 and authors and venue in conf_list:
                min_year = min(min_year, year)
                max_year = max(max_year, year)
                authors = authors.split(',')
                terms = parse_term(title)
                n_authors = len(authors)
                n_terms = len(terms)
                published += 1

                p = Paper(year)
                if feature_begin < year <= feature_end:
                    p.id = indexer.index('paper', id)
                    p.terms = [indexer.index('term', term) for term in terms]
                    p.references = [indexer.index('paper', paper_id) for paper_id in references]
                    p.authors = [indexer.index('author', author_name) for author_name in authors]
                    p.venue = indexer.index('venue', venue)
                    bisect.insort(papers_feature_window, p)
                elif observation_begin < year <= observation_end:
                    p.references = references
                    p.authors = authors
                    papers_observation_window.append(p)

        index, authors, title, year, venue = None, None, None, None, None
        references = []

        for p in papers_observation_window:
            authors = []
            references = []
            for author in p.authors:
                author_id = indexer.get_index('author', author)
                if author_id is not None:
                    authors.append(author_id)
            for ref in p.references:
                paper_id = indexer.get_index('paper', ref)
                if paper_id is not None:
                    references.append(paper_id)
            p.authors = authors
            p.references = references

    with open('data/metadata_%s.txt' % path, 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Authors: %d\n' % indexer.indices['author'])
        output.write('#Papers: %d\n' % indexer.indices['paper'])
        output.write('#Venues: %d\n' % indexer.indices['venue'])
        output.write('#Terms: %d\n\n' % indexer.indices['term'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Write: %d\n' % n_authors)
        output.write('#Cite: %d\n' % n_citation)
        output.write('#Publish: %d\n' % published)
        output.write('#Contain: %d\n' % n_terms)
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % min_year)
        output.write('To: %s\n' % max_year)

    result = papers_feature_window, papers_observation_window, indexer.indices
    # pickle.dump(result, open('data/papers_%s.pkl' % path, 'wb'))
    return result


def run(delta, observation_window, n_snapshots):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_path = os.getcwd()
    os.chdir(dir_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(message)s',datefmt='%H:%M:%S')

    conf_list = {
        'db': [
            'KDD', 'PKDD', 'ICDM', 'SDM', 'PAKDD', 'SIGMOD', 'VLDB', 'ICDE', 'PODS', 'EDBT', 'SIGIR', 'ECIR',
            'ACL', 'WWW', 'CIKM', 'NIPS', 'ICML', 'ECML', 'AAAI', 'IJCAI'
        ],

        'th': [
            'STOC', 'FOCS', 'COLT', 'LICS', 'SCG', 'SODA', 'SPAA', 'PODC', 'ISSAC', 'CRYPTO', 'EUROCRYPT', 'CONCUR',
            'ICALP', 'STACS', 'COCO', 'WADS', 'MFCS', 'SWAT', 'ESA', 'IPCO', 'LFCS', 'ALT', 'EUROCOLT', 'WDAG',
            'ISTCS', 'FSTTCS', 'LATIN', 'RECOMB', 'CADE', 'ISIT', 'MEGA', 'ASIAN', 'CCCG', 'FCT', 'WG', 'CIAC', 'ICCI',
            'CATS', 'COCOON', 'GD', 'ISAAC', 'SIROCCO', 'WEA', 'ALENEX', 'FTP', 'CSL', 'DMTCS'
        ]
    }

    observation_end = 2017
    observation_begin = observation_end - observation_window
    feature_end = observation_begin
    featreu_begin = feature_end - delta * n_snapshots

    papers_feat_window, papers_obs_window, counter = genereate_papers('', featreu_begin, feature_end,
                                                                     observation_begin, observation_end,
                                                                     conf_list)



    if __name__ == '__main__':
        run()


