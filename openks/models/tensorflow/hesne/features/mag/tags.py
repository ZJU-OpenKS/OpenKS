# -*- coding: UTF-8 -*-
import sys
import chardet
sys.path.append('..')
from itertools import islice
import pickle as pkl
from utils import Indexer
import json
import time
mag_root = '/home1/wyf/Projects/dynamic_network_embedding/data/MAG2/papers/mag_papers_'
venue_root = '/home1/wyf/Projects/dynamic_network_embedding/data/MAG2/venues/mag_venues.txt'
venue_choosed = '/home1/wyf/Projects/dynamic_network_embedding/code/features/mag/venue_list_choosed.txt'
venue_choosed_lowercase = '/home1/wyf/Projects/dynamic_network_embedding/code/features/mag/venue_list_choosed_normalize.txt'
paper_choosed = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/paper_choosed.txt'

def mag_normalize():
    venue_dict = {}
    venue_choosed_dict = {}
    # venue_choosedid_dict = {}
    with open(venue_root, 'r') as fvenue:
        for line in fvenue:
            venue_data = json.loads(line)
            if 'ConferenceId' in venue_data.keys():
                venue_dict[venue_data['DisplayName']] = venue_data['ConferenceId']
            if 'JournalId' in venue_data.keys():
                venue_dict[venue_data['DisplayName']] = venue_data['JournalId']
    with open(venue_choosed, 'r', encoding='ascii') as fvenue_choosed:
        with open(venue_choosed_lowercase, 'w') as fvenue_lower:
            for line in fvenue_choosed:
                value = line.strip().split('\t')
                print(value)
                venue_display = value[0]
                venue_id = venue_dict[venue_display]
                venue = value[1]
                venue_display = venue_display.lower()
                # venue_choosed_dict[venue_display] = venue
                venue_choosed_dict[venue_id] = [venue_display]
                fvenue_lower.write(venue_id+'\t'+venue_display+'\t'+venue+'\n')
    return venue_choosed_dict

def choose_mag():
    venue_dict = mag_normalize()
    paper_dict = {}
    author_dict = {}
    keyword_dict = {}
    with open(paper_choosed, 'w') as f_paper:
        for i in range(10):
            with open(mag_root+str(i)+'.txt', 'r') as f_mag:
                for line in f_mag:
                    paper_data = json.loads(line)
                    if 'venue' not in paper_data.keys():
                        continue
                    venue_data = paper_data['venue']
                    if 'id' not in venue_data.keys():
                        continue
                    else:
                        if venue_data['id'] not in venue_dict.keys():
                            continue
                        else:
                            venue = venue_dict[venue_data['id']]
                    if 'id' not in paper_data.keys():
                        continue
                    if 'title' not in paper_data.keys():
                        continue
                    if 'keywords' not in paper_data.keys():
                        continue
                    if len(paper_data['keywords'])<1:
                        continue
                    if 'references' not in paper_data.keys():
                        continue
                    if len(paper_data['references'])<1:
                        continue
                    if 'year' not in paper_data.keys():
                        continue
                    if 'authors' not in paper_data.keys():
                        continue
                    author_ids = []
                    for author in paper_data['authors']:
                        if 'id' not in author.keys():
                            continue
                        author_id = author['id']
                        author_ids.append(author_id)
                    if len(author_ids) < 1:
                        continue
                        # author_dict[author_id] = author_dict.get(author_id, 0)+1
                    event = {}
                    event['id'] = paper_data['id']
                    paper_dict[event['id']] = paper_dict.get(event['id'], 0)+1
                    event['title'] = paper_data['title']
                    event['venue'] = venue
                    event['author'] = author_ids
                    for author_id in author_ids:
                        author_dict[author_id] = author_dict.get(author_id, 0)+1
                    event['keyword'] = paper_data['keywords']
                    for keyword in event['keyword']:
                        keyword_dict[keyword] = keyword_dict.get(keyword, 0)+1
                    event['reference'] = paper_data['references']
                    event['year'] = str(paper_data['year'])
                    f_paper.write(event['id']+'\t'+event['title']+'\t'+event['venue']+'\t'+';'.join(event['author'])+'\t'+\
                            ';'.join(event['keyword'])+'\t'+';'.join(event['reference'])+'\t'+event['year']+'\n')
    print(len(paper_dict.keys()))
    print(len(author_dict.keys()))
    print(len(keyword_dict.keys()))

if __name__ == '__main__':
    choose_mag()


