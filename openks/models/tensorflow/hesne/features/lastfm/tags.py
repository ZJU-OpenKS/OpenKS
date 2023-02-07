# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')
from itertools import islice
import pickle as pkl
from utils import Indexer
import time
user_taggedartists_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/hetrec2011/user_taggedartists-timestamps.dat'
user_taggedartists_sort_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/processed/user_taggedartists-sort-timestamps.dat'
user_taggedartists_concat_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/processed/user_taggedartists-concat-timestamps.dat'
# event_changed_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/event_out.txt'
event_sorted_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/processed/event_sorted.txt'
indexer_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/processed/indexer.pkl'
# user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/user_id.pkl'
# artist_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/artist_id.pkl'
# tag_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/lastfm/tag_id.pkl'


def choose_taggedartists():
    user_dict = {}
    artist_dict = {}
    tag_dict = {}
    with open(user_taggedartists_root, 'r') as tag_data:
        for line in islice(tag_data, 1, None):
            value = line.strip().split('\t')
            user = value[0]
            artist = value[1]
            tag = value[2]
            user_dict[user] = user_dict.get(user,0) + 1
            artist_dict[artist] = artist_dict.get(artist,0) + 1
            tag_dict[tag] = tag_dict.get(tag,0) + 1
        print(len(user_dict.keys()))
        print(len(artist_dict.keys()))
        print(len(tag_dict.keys()))
        user_list = [u for u in user_dict.keys() if user_dict[u]>1]
        artist_list = [b for b in artist_dict.keys() if artist_dict[b]>1]
        tag_list = [t for t in tag_dict.keys() if tag_dict[t]>1]
        print(len(user_list))
        print(len(artist_list))
        print(len(tag_list))
    return user_list, artist_list, tag_list


def sort_byuser():
    user_event_list = []
    num = 0
    with open(user_taggedartists_root, 'r') as tag_data:
        with open(user_taggedartists_sort_root, 'w') as tag_sort_data:
            for line in islice(tag_data, 1, None):
                value = line.strip().split('\t')
                user = value[0]
                artist = value[1]
                tag = value[2]
                timestamp = value[3]
                event={}
                event['user'] = user
                event['artist'] = artist
                event['tag'] = tag
                event['timestamp'] = timestamp
                if num == 0:
                    lastuser = user
                if lastuser != user:
                    user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
                    for user_event in user_event_list_sorted:
                        tag_sort_data.write(str(user_event['user'])+'\t'+str(user_event['artist'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')
                    lastuser = user
                    user_event_list = [event]
                else:
                    user_event_list.append(event)
                num+=1
            user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
            for user_event in user_event_list_sorted:
                tag_sort_data.write(str(user_event['user'])+'\t'+str(user_event['artist'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')

def concat_taggedartists(user_list, artist_list, tag_list):
    sort_byuser()
    taglist = []
    num = 0
    user_dict = {}
    artist_dict = {}
    tag_dict = {}
    with open(user_taggedartists_sort_root, 'r') as tag_data:
        with open(user_taggedartists_concat_root, 'w') as tag_concat_data:
            for line in tag_data:
                value = line.strip().split('\t')
                user = value[0]
                artist = value[1]
                tag = value[2]
                if user not in user_list:
                    continue
                if artist not in artist_list:
                    continue
                if tag not in tag_list:
                    continue
                user_dict[user] = user_dict.get(user, 0) + 1
                artist_dict[artist] = artist_dict.get(artist, 0) + 1
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
                timestamp = value[3]
                tagcontext = str(user)+'_'+str(artist)+'_'+str(timestamp)
                if num == 0:
                    lasttagcontext = tagcontext
                if tagcontext != lasttagcontext:
                    value_concat = lasttagcontext.split('_')
                    user_concat = value_concat[0]
                    artist_concat = value_concat[1]
                    timestamp_concat = value_concat[2]
                    tag_concat = ';'.join(taglist)
                    tag_concat_data.write(user_concat+'\t'+artist_concat+'\t'+tag_concat+'\t'+timestamp_concat+'\n')
                    lasttagcontext = tagcontext
                    taglist = [str(tag)]
                else:
                    taglist.append(str(tag))
                num+=1
            value_concat = lasttagcontext.split('_')
            user_concat = value_concat[0]
            artist_concat = value_concat[1]
            timestamp_concat = value_concat[2]
            tag_concat = ';'.join(taglist)
            tag_concat_data.write(user_concat+'\t'+artist_concat+'\t'+tag_concat+'\t'+timestamp_concat+'\n')
    return user_dict.keys(), artist_dict.keys(), tag_dict.keys()

# def change_eventwithid(user_list, artist_list, tag_list):
#     indexer = Indexer({'user': 0, 'bookmark': len(user_list), 'tag': len(user_list) + len(bookmark_list)})
#     #############tag bookmark#########
#     user_dict = {}
#     artist_dict = {}
#     tag_dict = {}
#     with open(user_taggedartists_concat_root, 'r') as tag_concat_data:
#         with open(event_changed_root, 'w') as changeid_out:
#             for line in tag_concat_data:
#                 value = line.split('\t')
#                 user = value[0]
#                 artist = value[1]
#                 tags = value[2]
#                 event_tags = tags.split(';')
#                 timestamp = value[3].replace('\n', '')
#                 if user not in user_dict.keys():
#                     user_dict[user] = len(user_dict.keys())
#                 if artist not in artist_dict.keys():
#                     artist_dict[artist] = len(artist_dict.keys())
#                 changed_eventtags = []
#                 for tag in event_tags:
#                     if tag not in tag_dict:
#                         tag_dict[tag] = len(tag_dict.keys())
#                         changed_eventtags.append(str(tag_dict[tag]))
#                     else:
#                         changed_eventtags.append(str(tag_dict[tag]))
#
#                 changeid_out.write(str(user_dict[user])+'\t'+str(artist_dict[artist])+'\t'+';'.join(changed_eventtags)+'\t'+str(timestamp)+'\n')
#     with open(user_id_root, 'wb') as user_id:
#         pkl.dump(user_dict, user_id)
#     with open(artist_id_root, 'wb') as artist_id:
#         pkl.dump(artist_dict, artist_id)
#     with open(tag_id_root, 'wb') as tag_id:
#         pkl.dump(tag_dict, tag_id)
#
# def sort_event():
#     event_list = []
#     with open(event_changed_root, 'r') as changed_event:
#         with open(event_sorted_root, 'w') as event_sorted:
#             for line in changed_event:
#                 event = {}
#                 value = line.split('\t')
#                 event['user'] = value[0]
#                 event['artist'] = value[1]
#                 event['tag'] = value[2]
#                 event['timestamp'] = value[3].replace('\n','')
#                 event_list.append(event)
#             sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
#             for event in sorted_list:
#                 event_sorted.write(str(event['timestamp']) + '\t' + str(event['user']) + '\t' + str(event['artist']) + '\t'
#                                    + str(event['tag']) + '\n')

def change_taggedartists(user_list, artist_list, tag_list):
    indexer = Indexer({'user':0, 'artist':len(user_list), 'tag':len(user_list)+len(artist_list)})
    #############tag bookmark#########
    event_list = []
    event_num = 0.0
    event_tag_sum = 0.0
    with open(user_taggedartists_concat_root, 'r') as tag_concat_data:
        for line in tag_concat_data:
            event = {}
            event_num += 1
            value = line.strip().split('\t')
            user = value[0]
            artist = value[1]
            tags = value[2].split(';')
            timestamp = value[3]
            indexer.index('user', user)
            indexer.index('artist', artist)
            for tag in tags:
                indexer.index('tag', tag)
            event['user'] = indexer.get_index('user', user)
            event['artist'] = indexer.get_index('artist', artist)
            event['tag'] = ';'.join([str(indexer.get_index('tag', tag)) for tag in tags])
            event['timestamp'] = float(timestamp)/1000
            event_tag_sum += len(tags)
            event_list.append(event)
    event_sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
    print(event_tag_sum / event_num)
    print(event_tag_sum)
    print(event_num)
    with open(event_sorted_root, 'w') as event_sorted:
        for event in event_sorted_list:
            event_sorted.write(str(event['timestamp']) + '\t' + str(event['user']) + '\t' + str(event['artist']) + '\t'
                + str(event['tag']) + '\n')
    with open(indexer_root, 'wb') as indexer_data:
        pkl.dump(indexer, indexer_data)


if __name__ == '__main__':
    user_list, artist_list, tag_list = choose_taggedartists()
    user_list, artist_list, tag_list = concat_taggedartists(user_list, artist_list, tag_list)
    print(len(user_list))
    print(len(artist_list))
    print(len(tag_list))
    # 1810 10753 4372
    change_taggedartists(user_list, artist_list, tag_list)
    # change_eventwithid()
    # sort_event()
