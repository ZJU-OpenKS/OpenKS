# Copyright (c) 2021 OpenKS Authors, Dlib Lab, Peking University. 
# All Rights Reserved.

from sknetwork.ranking import PageRank
import sknetwork
from ...model import TorchModel

@TorchModel.register("RelavanceScore", "PyTorch")
class RelavanceScore(TorchModel):
    '''
    Calculate relavance score between countries and concepts
    Return a N*M matrix
    N: country num
    M: concept num
    '''
    def __init__(self):
        super(RelavanceScore, self).__init__()
        self.pagerank = PageRank()
        
    def run(self, adj, seeds, paper_id, country_id, concept_id, paper_country, paper_concept):
        pr_scores = self.pagerank.fit_transform(adj, seeds) # pagerank scores
        w_paper = pr_scores[0:len(paper_id)]
        w_paper /= w_paper.sum() # normalize the paper weight
        
        # calculate the paper-county relavance score
        paper_country_edge = []
        country = set()
        for p in paper_country:
            for c in paper_country[p]:
                country.add(c)
                paper_country_edge.append((paper_id[p], \
                                           country_id[c] - (len(paper_id) + len(concept_id)), 
                                           1 / len(paper_country[p])))
        # add the countries without papers and set their edges to 0 
        for c in country_id:
            if c not in country:
                paper_country_edge.append((paper_id[0], \
                                           country_id[c] - (len(paper_id) + len(concept_id)), 0))
        country_paper_mat = sknetwork.utils.edgelist2biadjacency(paper_country_edge).transpose()
        
        # calculate the paper-concept relavance score
        paper_concept_edge = []
        concept = set()
        for p in paper_country:
            # add the paper without concepts and set their edges to 0
            if p not in paper_concept:
                paper_concept_edge.append((paper_id[p], concept_id[72] - (len(paper_id)), 0))
                continue
            for c in paper_concept[p]:
                paper_concept_edge.append((paper_id[p], concept_id[c] - (len(paper_id)), \
                                           1 / len(paper_concept[p]) * w_paper[paper_id[p]]))
                concept.add(c)
        # add the concepts not belonging to any papers and set their edges to 0
        for c in concept_id:
            if c not in concept:
                paper_concept_edge.append((paper_id[0], concept_id[c] - (len(paper_id)), 0))
        paper_concept_mat = sknetwork.utils.edgelist2biadjacency(paper_concept_edge)
        country_concept = country_paper_mat.dot(paper_concept_mat)
        return country_concept
