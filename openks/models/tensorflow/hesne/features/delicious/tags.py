# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')
import pickle as pkl
from itertools import islice
from utils import Indexer

user_taggedbookmarks_root = '/home1/wyf/Projects/dynamic_network_embedding/data/delicious/hetrec2011/user_taggedbookmarks-timestamps.dat'
user_taggedbookmarks_sort_root = '/home1/wyf/Projects/dynamic_network_embedding/data/delicious/processed/user_taggedbookmarks-sort-timestamps.dat'
user_taggedbookmarks_concat_root = '/home1/wyf/Projects/dynamic_network_embedding/data/delicious/processed/user_taggedbookmarks_concat-timestamps.dat'
event_sorted_root = '/home1/wyf/Projects/dynamic_network_embedding/data/delicious/processed/event_sorted.txt'
indexer_root = '/home1/wyf/Projects/dynamic_network_embedding/data/delicious/processed/indexer.pkl'

def choose_taggedbookmarks():
    user_dict = {}
    bookmark_dict = {}
    tag_dict = {}
    with open(user_taggedbookmarks_root, 'r') as tag_data:
        for line in islice(tag_data, 1, None):
            value = line.strip().split('\t')
            user = value[0]
            bookmark = value[1]
            tag = value[2]
            user_dict[user] = user_dict.get(user,0) + 1
            bookmark_dict[bookmark] = bookmark_dict.get(bookmark,0) + 1
            tag_dict[tag] = tag_dict.get(tag,0) + 1
        print(len(user_dict.keys()))
        print(len(bookmark_dict.keys()))
        print(len(tag_dict.keys()))
        user_list = [u for u in user_dict.keys() if user_dict[u]>5]
        bookmark_list = [b for b in bookmark_dict.keys() if bookmark_dict[b]>5]
        tag_list = [t for t in tag_dict.keys() if tag_dict[t]>5]
    return user_list, bookmark_list, tag_list
    # print(len(userDict))
    # print(len(bookmarkDict))
    # print(len(tagDict))

def sort_byuser():
    user_event_list = []
    num = 0
    with open(user_taggedbookmarks_root, 'r') as tag_data:
        with open(user_taggedbookmarks_sort_root, 'w') as tag_sort_data:
            for line in islice(tag_data, 1, None):
                value = line.strip().split('\t')
                user = value[0]
                bookmark = value[1]
                tag = value[2]
                timestamp = value[3]
                event={}
                event['user'] = user
                event['bookmark'] = bookmark
                event['tag'] = tag
                event['timestamp'] = timestamp
                if num == 0:
                    lastuser = user
                if lastuser != user:
                    user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
                    for user_event in user_event_list_sorted:
                        tag_sort_data.write(str(user_event['user'])+'\t'+str(user_event['bookmark'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')
                    lastuser = user
                    user_event_list = [event]
                else:
                    user_event_list.append(event)
                num+=1
            user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
            for user_event in user_event_list_sorted:
                tag_sort_data.write(str(user_event['user'])+'\t'+str(user_event['bookmark'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')

def concat_taggedbookmarks(user_list, bookmark_list, tag_list):
    sort_byuser()
    taglist = []
    num = 0
    user_dict = {}
    bookmark_dict = {}
    tag_dict = {}
    with open(user_taggedbookmarks_sort_root, 'r') as tag_data:
        with open(user_taggedbookmarks_concat_root, 'w') as tag_concat_data:
            for line in tag_data:
                value = line.strip().split('\t')
                user = value[0]
                bookmark = value[1]
                tag = value[2]
                if user not in user_list:
                    continue
                if bookmark not in bookmark_list:
                    continue
                if tag not in tag_list:
                    continue
                user_dict[user] = user_dict.get(user, 0) + 1
                bookmark_dict[bookmark] = bookmark_dict.get(bookmark, 0) + 1
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
                timestamp = value[3]
                tagcontext = str(user)+'_'+str(bookmark)+'_'+str(timestamp)
                if num == 0:
                    lasttagcontext = tagcontext
                if tagcontext != lasttagcontext:
                    value_concat = lasttagcontext.split('_')
                    user_concat = value_concat[0]
                    bookmark_concat = value_concat[1]
                    timestamp_concat = value_concat[2]
                    tag_concat = ';'.join(taglist)
                    tag_concat_data.write(user_concat+'\t'+bookmark_concat+'\t'+tag_concat+'\t'+timestamp_concat+'\n')
                    lasttagcontext = tagcontext
                    taglist = [str(tag)]
                else:
                    taglist.append(str(tag))
                num+=1
            value_concat = lasttagcontext.split('_')
            user_concat = value_concat[0]
            bookmark_concat = value_concat[1]
            timestamp_concat = value_concat[2]
            tag_concat = ';'.join(taglist)
            tag_concat_data.write(user_concat+'\t'+bookmark_concat+'\t'+tag_concat+'\t'+timestamp_concat+'\n')
    return user_dict.keys(), bookmark_dict.keys(), tag_dict.keys()

def change_taggedbookmarks(user_list, bookmark_list, tag_list):
    indexer = Indexer({'user':0, 'bookmark':len(user_list), 'tag':len(user_list)+len(bookmark_list)})
    #############tag bookmark#########
    event_list = []
    event_num = 0.0
    event_tag_sum = 0.0
    with open(user_taggedbookmarks_concat_root, 'r') as tag_concat_data:
        for line in tag_concat_data:
            event = {}
            event_num += 1
            value = line.strip().split('\t')
            user = value[0]
            bookmark = value[1]
            tags = value[2].split(';')
            timestamp = value[3]
            indexer.index('user', user)
            indexer.index('bookmark', bookmark)
            for tag in tags:
                indexer.index('tag', tag)
            event['user'] = indexer.get_index('user', user)
            event['bookmark'] = indexer.get_index('bookmark', bookmark)
            event['tag'] = ';'.join([str(indexer.get_index('tag', tag)) for tag in tags])
            event_tag_sum += len(tags)
            event['timestamp'] = float(timestamp)/1000
            event_list.append(event)
    print(event_tag_sum/event_num)
    print(event_tag_sum)
    print(event_num)
    event_sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
    with open(event_sorted_root, 'w') as event_sorted:
        for event in event_sorted_list:
            event_sorted.write(str(event['timestamp']) + '\t' + str(event['user']) + '\t' + str(event['bookmark']) + '\t'
                + str(event['tag']) + '\n')
    with open(indexer_root, 'wb') as indexer_data:
        pkl.dump(indexer, indexer_data)

if __name__ == '__main__':
    user_list, bookmark_list, tag_list = choose_taggedbookmarks()
    print(len(user_list))
    print(len(bookmark_list))
    print(len(tag_list))
    user_list, bookmark_list, tag_list = concat_taggedbookmarks(user_list, bookmark_list, tag_list)
    print(len(user_list))
    print(len(bookmark_list))
    print(len(tag_list))
    #1768 28716 7900
    change_taggedbookmarks(user_list, bookmark_list, tag_list)
