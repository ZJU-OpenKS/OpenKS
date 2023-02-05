import os
import math
import pickle as pkl
import random
# user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/user_small_id.txt'
# business_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business_small_id.pkl'
# term_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/term_small_id.pkl'
# review_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review_small_id.pkl'
# user_embedding_outroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/user_small_embedding.pkl'
# business_embedding_outroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business_small_embedding.pkl'
# term_embedding_outroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/term_small_embedding.pkl'
def load_event_data(event_data_root, data_set):
    event_list = []
    with open(event_data_root, 'r') as event_data:
        for line in event_data:
            # event = {'type': '', 'event_nodes': []}
            event = []
            value = line.strip().split('\t')
            # if data_set == 'yelp':
            #     # star = value[1]
            #     user = value[3]
            #     business = value[4]
            #     terms = value[5].split(';')
            #     # event['type'] = star
            #     # event['event_nodes'].append([int(user)])
            #     # event['event_nodes'].append([int(business)])
            #     # event['event_nodes'].append([int(term) for term in terms])
            #     event.append([int(user)])
            #     event.append([int(business)])
            #     event.append([int(term) for term in terms])
            if data_set == 'delicious':
                timestamp = value[0]
                user = value[1]
                bookmark = value[2]
                tags = value[3].split(';')
                tags_int = [int(tag) for tag in tags]
                tags_int.sort()
                subevent = []
                for tag in tags_int:
                    subevent.append([int(user), int(bookmark), tag])
                event.append([int(user)])
                event.append([int(bookmark)])
                event.append(tags_int)
                event.append(float(timestamp))
                event.append(subevent)
            elif data_set == 'lastfm':
                timestamp = value[0]
                user = value[1]
                artist = value[2]
                tags = value[3].split(';')
                tags_int = [int(tag) for tag in tags]
                tags_int.sort()
                subevent = []
                for tag in tags_int:
                    subevent.append([int(user), int(artist), tag])
                event.append([int(user)])
                event.append([int(artist)])
                event.append(tags_int)
                event.append(float(timestamp))
                event.append(subevent)
            elif data_set == 'movielens':
                timestamp = value[0]
                user = value[1]
                movie = value[2]
                tags = value[3].split(';')
                genres = value[4].split(';')
                directors = value[5].split(';')
                actors = value[6].split(';')
                countries = value[7].split(';')
                tags_int = [int(tag) for tag in tags]
                genres_int = [int(genre) for genre in genres]
                directors_int = [int(director) for director in directors]
                actors_int = [int(actor) for actor in actors]
                countries_int = [int(country) for country in countries]
                genres_int.sort()
                directors_int.sort()
                actors_int.sort()
                countries_int.sort()
                subevent = []
                for tag in tags_int:
                    for genre in genres_int:
                        for director in directors_int:
                            for actor in actors_int:
                                for country in countries_int:
                                    subevent.append([int(user), int(movie), tag, genre, director, actor, country])
                event.append([int(user)])
                event.append([int(movie)])
                event.append(tags_int)
                event.append(genres_int)
                event.append(directors_int)
                event.append(actors_int)
                event.append(countries_int)
                event.append(float(timestamp))
                event.append(subevent)
            elif data_set == 'mag':
                timestamp = value[0]
                venue = value[1]
                authors = value[3].split(';')
                keywords = value[4].split(';')
                authors_int = [int(author) for author in authors]
                keywords_int = [int(keyword) for keyword in keywords]
                authors_int.sort()
                keywords_int.sort()
                subevent = []
                for author in authors_int:
                    for keyword in keywords_int:
                        subevent.append([int(venue), author, keyword])
                event.append([int(venue)])
                event.append(authors_int)
                event.append(keywords_int)
                event.append(float(timestamp))
                event.append(subevent)
            else:
                print('dataset wrong')
                break
            event_list.append(event)
    return event_list


def save_event_data(event_data_root, batch_size):
    event_num = sum(1 for line in open(os.path.join(event_data_root, 'event_sorted.txt')))
    batch_num = event_num // batch_size
    train_batch_num = int(batch_num * 0.8)
    valid_batch_num = int(batch_num * 0.1)
    event_trainroot = os.path.join(event_data_root, 'event_traindata.txt')
    event_validroot = os.path.join(event_data_root, 'event_validdata.txt')
    event_testroot = os.path.join(event_data_root, 'event_testdata.txt')
    with open(os.path.join(event_data_root, 'event_sorted.txt'), 'r') as event_data:
        with open(event_trainroot, 'w') as train_data:
            with open(event_validroot, 'w') as valid_data:
                with open(event_testroot, 'w') as test_data:
                    cur_num = 0
                    for line in event_data:
                        if cur_num < int(train_batch_num*batch_size):
                            train_data.write(line)
                        elif int(train_batch_num*batch_size) <= cur_num < int((train_batch_num+valid_batch_num)*batch_size):
                            valid_data.write(line)
                        else:
                            test_data.write(line)
                        cur_num += 1

def get_degree(event_list, type_num):
    degrees = [{} for _ in range(type_num)]
    for event in event_list:
        for type in range(type_num):
            for node in event[type]:
                degrees[type][node] = degrees[type].get(node, 0) + 1
    return degrees


class BatchData(object):
    def __init__(self, params, event_list, start_index):
        self.params = params
        self.event_list = event_list
        self.start = 0
        self.start_index = start_index
        self.list_length = len(event_list)
        self.batch_size = self.params['batch_event_numbers']
        self.type_num = self.params['node_type_numbers']

    def next_batch(self):
        if self.start+self.batch_size<=self.list_length:
            start = self.start
            end = self.start+self.batch_size
        else:
            print('load batch mistake')
        self.start = self.start+int(self.batch_size)
        if end<self.list_length:
            epoch_flag = False
        else:
            self.start = 0
            epoch_flag = True
        #one for the timestamp, two for the subeventlist, three for the event_index
        batch = [[[] for _ in range(self.type_num+3)] for _ in range(self.batch_size)]
        for event_index in range(start, end):
            for type_index in range(len(self.event_list[event_index])):
                nodes = self.event_list[event_index][type_index]
                batch[event_index - start][type_index] = nodes
            batch[event_index - start][-1] = event_index+self.start_index
        return batch, epoch_flag

    def get_batch_num(self):
        return int(math.ceil(len(self.event_list) / float(self.batch_size)))

    def get_subevents(self):
        subevents = {}
        events_time = {}
        for event_index in range(len(self.event_list)):
            subevents[event_index+self.start_index] = self.event_list[event_index][-1]
            events_time[event_index+self.start_index] = self.event_list[event_index][-2]
        return subevents, events_time

    def shuffle(self):
        random.shuffle(self.event_list)


if '__name__' == '__main__':
    load_event_data()
