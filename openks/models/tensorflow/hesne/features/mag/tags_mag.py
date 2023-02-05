# -*- coding: UTF-8 -*-
import sys
# import chardet
import time
sys.path.append('..')
from itertools import islice
import pickle as pkl
from utils import Indexer
import json

mag_root = '/home1/wyf/Projects/dynamic_network_embedding/data/MAG/mag_papers/mag_papers_'
venue_choosed = '/home1/wyf/Projects/dynamic_network_embedding/code/features/mag/venue_list_choosed.txt'
venue_choosed_lowercase = '/home1/wyf/Projects/dynamic_network_embedding/code/features/mag/venue_list_choosed_normalize.txt'
paper_choosed = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/paper_choosed.txt'
paper_choosed_small = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/paper_choosed_small.txt'
event_sorted_root = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/processed/event_sorted.txt'
indexer_root = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/processed/indexer.pkl'
event_indexer_root = '/home1/wyf/Projects/dynamic_network_embedding/data/mag/processed/event_indexer.pkl'

def mag_normalize():
    venue_choosed_dict = {}
    with open(venue_choosed, 'r', encoding='ascii') as fvenue_choosed:
        with open(venue_choosed_lowercase, 'w') as fvenue_lower:
            for line in fvenue_choosed:
                value = line.strip().split('\t')
                print(value)
                venue_display = value[0]
                venue = value[1]
                venue_display = venue_display.lower()
                # venue_choosed_dict[venue_display] = venue
                venue_choosed_dict[venue_display] = [venue]
                fvenue_lower.write(venue_display+'\t'+venue+'\n')
    return venue_choosed_dict

def choose_mag():
    venue_dict = mag_normalize()
    paper_dict = {}
    author_dict = {}
    keyword_dict = {}
    with open(paper_choosed, 'w') as f_paper:
        for i in range(167):
            with open(mag_root+str(i)+'.txt', 'r') as f_mag:
                for line in f_mag:
                    paper_data = json.loads(line)
                    if 'venue' not in paper_data.keys():
                        continue
                    if paper_data['venue'] not in venue_dict.keys():
                        continue
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
                    author_names = []
                    for author in paper_data['authors']:
                        author_name = author['name']
                        author_names.append(author_name)
                    if len(author_names) < 1:
                        continue
                    event = {}
                    event['id'] = paper_data['id']
                    paper_dict[event['id']] = paper_dict.get(event['id'], 0)+1
                    event['title'] = paper_data['title']
                    event['venue'] = paper_data['venue']
                    event['author'] = author_names
                    for author_name in author_names:
                        author_dict[author_name] = author_dict.get(author_name, 0)+1
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

def choose_small():
    paper_dict = {}
    author_dict = {}
    keyword_dict = {}
    with open(paper_choosed, 'r') as f_choosed:
        for line in f_choosed:
            value = line.strip().split('\t')
            paper = value[0]
            authors = value[3].split(';')
            keywords = value[4].split(';')
            paper_dict[paper] = paper_dict.get(paper, 0) + 1
            for author in authors:
                author_dict[author] = author_dict.get(author, 0) + 1
            for keyword in keywords:
                keyword_dict[keyword] = keyword_dict.get(keyword, 0) + 1
    paper_list = set([p for p in paper_dict.keys()])
    author_list = set([a for a in author_dict.keys() if author_dict[a]>0])
    keyword_list = set([k for k in keyword_dict.keys() if keyword_dict[k]>0])
    print(len(paper_list))
    print(len(author_list))
    print(len(keyword_list))

    # tune = 1
    # flag = True
    # while(flag):
    #     paper_choosed_list = []
    #     author_choosed_list = []
    #     keyword_choosed_list = []
    #     with open(paper_choosed, 'r') as f_choosed:
    #         for line in f_choosed:
    #             value = line.strip().split('\t')
    #             paper = value[0]
    #             authors = value[3].split(';')
    #             keywords = value[4].split(';')
    #             references = value[5].split(';')
    #             if paper not in paper_list:
    #                 continue
    #             author_choosed = []
    #             for author in authors:
    #                 if author not in author_list:
    #                     continue
    #                 else:
    #                     author_choosed.append(author)
    #             if len(author_choosed)<1:
    #                 continue
    #             keyword_choosed = []
    #             for keyword in keywords:
    #                 if keyword not in keyword_list:
    #                     continue
    #                 else:
    #                     keyword_choosed.append(keyword)
    #             if len(keyword_choosed)<1:
    #                 continue
    #             reference_choosed = []
    #             for reference in references:
    #                 if reference not in paper_list:
    #                     continue
    #                 else:
    #                     reference_choosed.append(reference)
    #             if len(reference_choosed)<1:
    #                 continue
    #             paper_choosed_list.append(paper)
    #             author_choosed_list.extend(author_choosed)
    #             keyword_choosed_list.extend(keyword_choosed)
    #     paper_choosed_list = set(paper_choosed_list)
    #     author_choosed_list = set(author_choosed_list)
    #     keyword_choosed_list = set(keyword_choosed_list)
    #     if ((paper_choosed_list==paper_list) and (author_choosed_list==author_list) and (keyword_choosed_list==keyword_list)):
    #         flag = False
    #         paper_list = paper_choosed_list
    #         author_list = author_choosed_list
    #         keyword_list = keyword_choosed_list
    #     else:
    #         paper_list = paper_choosed_list
    #         author_list = author_choosed_list
    #         keyword_list = keyword_choosed_list
    #         print('tune '+str(tune))
    #         tune += 1
    #         print(len(paper_choosed_list))
    #         print(len(author_choosed_list))
    #         print(len(keyword_choosed_list))
    paper_choosed_list = set([])
    author_choosed_list = set([])
    keyword_choosed_list = set([])
    with open(paper_choosed, 'r') as f_choosed:
        with open(paper_choosed_small, 'w') as f_choosed_small:
            for line in f_choosed:
                value = line.strip().split('\t')
                paper = value[0]
                title = value[1]
                venue = value[2]
                authors = value[3].split(';')
                keywords = value[4].split(';')
                # references = value[5].split(';')
                timestamp = int(value[6])
                if paper not in paper_list:
                    continue
                author_choosed = []
                for author in authors:
                    if author not in author_list:
                        continue
                    else:
                        author_choosed.append(author)
                if len(author_choosed) < 1:
                    continue
                keyword_choosed = []
                for keyword in keywords:
                    if keyword not in keyword_list:
                        continue
                    else:
                        keyword_choosed.append(keyword)
                if len(keyword_choosed) < 1:
                    continue
                if timestamp < 2014:
                    continue
                # reference_choosed = []
                # for reference in references:
                #     if reference not in paper_list:
                #         continue
                #     else:
                #         reference_choosed.append(reference)
                # if len(reference_choosed) < 1:
                #     continue
                # f_choosed_small.write(paper+'\t'+title+'\t'+venue+'\t'+';'.join(author_choosed)+'\t'+';'.join(keyword_choosed)
                #                       +'\t'+';'.join(reference_choosed)+'\t'+str(timestamp)+'\n')
                f_choosed_small.write(paper + '\t' + title + '\t' + venue + '\t' + ';'.join(author_choosed) + '\t' + ';'.join(keyword_choosed)
                    + '\t' + str(timestamp) + '\n')
                paper_choosed_list.update([paper])
                author_choosed_list.update(author_choosed)
                keyword_choosed_list.update(keyword_choosed)
    return paper_choosed_list, author_choosed_list, keyword_choosed_list

def change_taggedmag(paper_choosed_list, author_choosed_list, keyword_choosed_list):
    indexer = Indexer({'venue':0, 'author':21, 'keyword':21+len(author_choosed_list)})
    print(len(paper_choosed_list))
    print(len(author_choosed_list))
    print(len(keyword_choosed_list))
    event_indexer = Indexer({'paper': 21+len(author_choosed_list)+len(keyword_choosed_list)})
    #####################tag mag####################
    event_list = []
    event_num = 0.0
    event_author_sum = 0.0
    event_keyword_sum = 0.0
    with open(paper_choosed_small, 'r') as f_choosed_small:
        for line in f_choosed_small:
            value = line.strip().split('\t')
            paper = value[0]
            venue = value[2]
            authors = value[3].split(';')
            keywords = value[4].split(';')
            # event_indexer.index('paper', paper)
            indexer.index('venue', venue)
            for author in authors:
                indexer.index('author', author)
            for keyword in keywords:
                indexer.index('keyword', keyword)
    # with open(paper_choosed_small, 'r') as f_choosed_small:
    #     for line in f_choosed_small:
            event = {}
            event_num += 1
    #         value = line.strip().split('\t')
    #         paper = value[0]
    #         venue = value[2]
    #         authors = value[3].split(';')
    #         keywords = value[4].split(';')
    #         # references = value[5].split(';')
            timestamp = value[5]
            # event['paper'] = event_indexer.get_index('paper', paper)
            event['paper'] = paper
            event['venue'] = indexer.get_index('venue', venue)
            event['author'] = ';'.join([str(indexer.get_index('author', author)) for author in authors])
            event['keyword'] = ';'.join([str(indexer.get_index('keyword', keyword)) for keyword in keywords])
            # event['reference'] = ';'.join([str(event_indexer.get_index('paper', reference)) for reference in references])
            event['timestamp'] = float(timestamp)
            event_author_sum += len(authors)
            event_keyword_sum += len(keywords)
            event_list.append(event)
    print(event_author_sum / event_num)
    print(event_keyword_sum / event_num)
    print(event_author_sum)
    print(event_keyword_sum)
    print(event_num)
    event_sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
    with open(event_sorted_root, 'w') as event_sorted:
        for event in event_sorted_list:
            event_indexer.index('paper', event['paper'])
            event['paper'] = event_indexer.get_index('paper', event['paper'])
            event_sorted.write(str(event['timestamp']) + '\t' + str(event['venue']) + '\t'+ str(event['paper']) + '\t' + \
                               event['author'] + '\t' + event['keyword'] + '\n')
    with open(indexer_root, 'wb') as indexer_data:
        pkl.dump(indexer, indexer_data)
    with open(event_indexer_root, 'wb') as indexer_data:
        pkl.dump(event_indexer, indexer_data)





# def print_small():
#     paper_dict = {}
#     author_dict = {}
#     keyword_dict = {}
#     with open(paper_choosed_small, 'r') as f_choosed_small:
#         for line in f_choosed_small:
#             value = line.strip().split('\t')
#             paper = value[0]
#             authors = value[3].split(';')
#             keywords = value[4].split(';')
#             paper_dict[paper] = paper_dict.get(paper, 0) + 1
#             for author in authors:
#                 author_dict[author] = author_dict.get(author, 0) + 1
#             for keyword in keywords:
#                 keyword_dict[keyword] = keyword_dict.get(keyword, 0) + 1
#     print(len(paper_dict.keys()))
#     print(len(author_dict.keys()))
#     print(len(keyword_dict.keys()))

if __name__ == '__main__':
    # choose_mag()
    paper_choosed_list, author_choosed_list, keyword_choosed_list = choose_small() #56200 paper 29006 author 33871 keywords
    change_taggedmag(paper_choosed_list, author_choosed_list, keyword_choosed_list)
    # print_small()


