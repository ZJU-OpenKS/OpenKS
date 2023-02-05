# -*- coding: UTF-8 -*-
import sys
sys.path.append('..')
from itertools import islice
import pickle as pkl
from utils import Indexer
import time
user_movietags_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/user_taggedmovies-timestamps.dat'
user_movietags_sort_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/processed/user_taggedmovies-sort-timestamps.dat'
user_movietags_concat_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/processed/user_taggedmovies-concat-timestamps.dat'
movie_genres_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/movie_genres.dat'
movie_directors_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/movie_directors.dat'
movie_actors_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/movie_actors.dat'
movie_countries_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/movie_countries.dat'
event_concat_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/processed/event_concat.dat'
# event_changed_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/event_out.txt'
event_sorted_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/processed/event_sorted.txt'
indexer_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/processed/indexer.pkl'
# user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/user_id.pkl'
# movie_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/movie_id.pkl'
# genre_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/genre_id.pkl'
# director_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/director_id.pkl'
# actor_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/actor_id.pkl'
# country_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/country_id.pkl'
# tag_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/movielens/hetrec2011/processed/tag_id.pkl'


def choose_taggedmovies():
    user_dict = {}
    movie_dict = {}
    tag_dict = {}
    with open(user_movietags_root, 'r') as tag_data:
        for line in islice(tag_data, 1, None):
            value = line.split('\t')
            user = value[0]
            movie = value[1]
            tag = value[2]
            user_dict[user] = user_dict.get(user,0) + 1
            movie_dict[movie] = movie_dict.get(movie,0) + 1
            tag_dict[tag] = tag_dict.get(tag,0) + 1
        print(len(user_dict.keys()))
        print(len(movie_dict.keys()))
        print(len(tag_dict.keys()))
        user_list = [u for u in user_dict.keys() if user_dict[u]>1]
        movie_list = [b for b in movie_dict.keys() if movie_dict[b]>1]
        tag_list = [t for t in tag_dict.keys() if tag_dict[t]>1]
        print(len(user_list))
        print(len(movie_list))
        print(len(tag_list))
    return user_list, movie_list, tag_list

def sort_byuser():
    user_event_list = []
    num = 0
    with open(user_movietags_root, 'r') as tag_data:
        with open(user_movietags_sort_root, 'w') as tag_sort_data:
            for line in islice(tag_data, 1, None):
                value = line.strip().split('\t')
                user = value[0]
                movie = value[1]
                tag = value[2]
                timestamp = value[3]
                event={}
                event['user'] = user
                event['movie'] = movie
                event['tag'] = tag
                event['timestamp'] = timestamp
                if num == 0:
                    lastuser = user
                if lastuser != user:
                    user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
                    for user_event in user_event_list_sorted:
                        tag_sort_data.write(str(user_event['user'])+'\t'+str(user_event['movie'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')
                    lastuser = user
                    user_event_list = [event]
                else:
                    user_event_list.append(event)
                num+=1
            user_event_list_sorted = sorted(user_event_list, key=lambda k: k['timestamp'])
            for user_event in user_event_list_sorted:
                tag_sort_data.write(
                    str(user_event['user'])+'\t'+str(user_event['movie'])+'\t'+str(user_event['tag'])+'\t'+str(user_event['timestamp'])+'\n')

def concat_taggedmovies(user_list, movie_list, tag_list):
    sort_byuser()
    taglist = []
    num = 0
    user_dict = {}
    movie_dict = {}
    tag_dict = {}
    with open(user_movietags_sort_root, 'r') as tag_data:
        with open(user_movietags_concat_root, 'w') as tag_concat_data:
            for line in tag_data:
                value = line.strip().split('\t')
                user = value[0]
                movie = value[1]
                tag = value[2]
                if user not in user_list:
                    continue
                if movie not in movie_list:
                    continue
                if tag not in tag_list:
                    continue
                user_dict[user] = user_dict.get(user, 0) + 1
                movie_dict[movie] = movie_dict.get(movie, 0) + 1
                tag_dict[tag] = tag_dict.get(tag, 0) + 1
                timestamp = value[3]
                tagcontext = str(user)+'_'+str(movie)+'_'+str(timestamp)
                if num == 0:
                    lasttagcontext = tagcontext
                if tagcontext != lasttagcontext:
                    value_concat = lasttagcontext.split('_')
                    user_concat = value_concat[0]
                    movie_concat = value_concat[1]
                    timestamp_concat = value_concat[2]
                    tag_concat = ';'.join(taglist)
                    tag_concat_data.write(user_concat+'\t'+movie_concat+'\t'+tag_concat+'\t'+timestamp_concat+'\n')
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

def concat_extrainfo():
    user_dict = {}
    movie_dict = {}
    tag_dict = {}
    genre_dict = {}
    director_dict = {}
    actor_dict = {}
    country_dict = {}
    movie_genres_dict = {}
    movie_directors_dict = {}
    movie_actors_dict = {}
    movie_countries_dict = {}
    f_movie_genres = open(movie_genres_root, 'r')
    f_movie_directors = open(movie_directors_root, 'r', encoding='latin-1')
    f_movie_actors = open(movie_actors_root, 'r', encoding='latin-1')
    f_movie_countries = open(movie_countries_root, 'r')
    for line in islice(f_movie_genres, 1, None):
        value = line.strip().split('\t')
        movie = value[0]
        genre = value[1]
        if movie not in movie_genres_dict.keys():
            movie_genres_dict[movie] = [genre]
        else:
            movie_genres_dict[movie].append(genre)
    for line in islice(f_movie_directors, 1, None):
        value = line.strip().split('\t')
        movie = value[0]
        director = value[1]
        if movie not in movie_directors_dict.keys():
            movie_directors_dict[movie] = [director]
        else:
            print('two director')
            movie_directors_dict[movie].append(director)
    for line in islice(f_movie_actors, 1, None):
        value = line.strip().split('\t')
        movie = value[0]
        actor = value[1]
        rank = int(value[3])
        if rank > 10:
            continue
        if movie not in movie_actors_dict.keys():
            movie_actors_dict[movie] = [actor]
        else:
            movie_actors_dict[movie].append(actor)
    for line in islice(f_movie_countries, 1, None):
        value = line.strip().split('\t')
        if len(value) == 1:
            continue
        movie = value[0]
        country = value[1]
        if movie not in movie_countries_dict.keys():
            movie_countries_dict[movie] = [country]
        else:
            print('two countries')
            movie_countries_dict[movie].append(country)
    f_movie_genres.close()
    f_movie_directors.close()
    f_movie_actors.close()
    f_movie_countries.close()
    with open(user_movietags_concat_root, 'r') as tag_concat_data:
        with open(event_concat_root, 'w') as event_concat_data:
            for line in tag_concat_data:
                value = line.strip().split('\t')
                user = value[0]
                movie = value[1]
                tags = value[2].split(';')
                timestamp = value[3]
                if movie not in movie_genres_dict.keys():
                    continue
                if movie not in movie_directors_dict.keys():
                    continue
                if movie not in movie_actors_dict.keys():
                    continue
                if movie not in movie_countries_dict.keys():
                    continue
                genres = movie_genres_dict[movie]
                directors = movie_directors_dict[movie]
                actors = movie_actors_dict[movie]
                countries = movie_countries_dict[movie]
                user_dict[user] = user_dict.get(user, 0) + 1
                movie_dict[movie] = movie_dict.get(movie, 0) + 1
                for tag in tags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1
                for genre in genres:
                    genre_dict[genre] = genre_dict.get(genre, 0) + 1
                for director in directors:
                    director_dict[director] = director_dict.get(director, 0) + 1
                for actor in actors:
                    actor_dict[actor] = actor_dict.get(actor, 0) + 1
                for country in countries:
                    country_dict[country] = country_dict.get(country, 0) + 1
                event_concat_data.write(str(user)+'\t'+str(movie)+'\t'
                                        +';'.join([str(tag) for tag in tags])+'\t'
                                        +';'.join([str(genre) for genre in genres])+'\t'
                                        +';'.join([str(director) for director in directors])+'\t'
                                        +';'.join([str(actor) for actor in actors])+'\t'
                                        +';'.join([str(country) for country in countries])+'\t'
                                        +str(timestamp)+'\n')
    # user_list = [u for u in user_dict.keys() if user_dict[u] > 1]
    # movie_list = [b for b in movie_dict.keys() if movie_dict[b] > 1]
    # tag_list = [t for t in tag_dict.keys() if tag_dict[t] > 1]
    # genre_list = [g for g in genre_dict.keys() if genre_dict[g] > 1]
    # director_list = [d for d in director_dict.keys() if director_dict[d] > 1]
    # actor_list = [a for a in actor_dict.keys() if actor_dict[a] > 1]
    # country_list = [c for c in country_dict.keys() if country_dict[c] > 1]
    return user_dict.keys(), movie_dict.keys(), tag_dict.keys(), genre_dict.keys(), director_dict.keys(), actor_dict.keys(), country_dict.keys()

def change_taggedmovies(user_list, movie_list, tag_list, genre_list, director_list, actor_list, country_list):
    indexer = Indexer({'user': 0, 'movie': len(user_list), 'tag': len(user_list) + len(movie_list),\
                       'genre': len(user_list) + len(movie_list) + len(tag_list),\
                       'director': len(user_list) + len(movie_list) + len(tag_list) + len(genre_list),\
                       'actor': len(user_list) + len(movie_list) + len(tag_list) + len(genre_list) + len(director_list),\
                       'country': len(user_list) + len(movie_list) + len(tag_list) + len(genre_list) + len(director_list) + len(actor_list)})
    ##########################
    event_list = []
    event_num = 0.0
    event_tag_sum = 0.0
    event_genre_sum = 0.0
    event_director_sum = 0.0
    event_actor_sum = 0.0
    event_country_sum = 0.0
    # user_dict = {}
    # movie_dict = {}
    # genre_dict = {}
    # director_dict = {}
    # actor_dict = {}
    # country_dict = {}
    # tag_dict = {}
    with open(event_concat_root, 'r') as tag_concat_data:
        for line in tag_concat_data:
            event = {}
            event_num += 1
            value = line.strip().split('\t')
            user = value[0]
            movie = value[1]
            tags = value[2].split(';')
            genres = value[3].split(';')
            directors = value[4].split(';')
            actors = value[5].split(';')
            countries = value[6].split(';')
            timestamp = value[7]

            indexer.index('user', user)
            indexer.index('movie', movie)
            for tag in tags:
                indexer.index('tag', tag)
            for genre in genres:
                indexer.index('genre', genre)
            for director in directors:
                indexer.index('director', director)
            for actor in actors:
                indexer.index('actor', actor)
            for country in countries:
                indexer.index('country', country)
            event['user'] = indexer.get_index('user', user)
            event['movie'] = indexer.get_index('movie', movie)
            event['tag'] = ';'.join([str(indexer.get_index('tag', tag)) for tag in tags])
            event['genre'] = ';'.join([str(indexer.get_index('genre', genre)) for genre in genres])
            event['director'] = ';'.join([str(indexer.get_index('director', director)) for director in directors])
            event['actor'] = ';'.join([str(indexer.get_index('actor', actor)) for actor in actors])
            event['country'] = ';'.join([str(indexer.get_index('country', country)) for country in countries])
            event['timestamp'] = float(timestamp)/1000
            event_tag_sum += len(tags)
            event_genre_sum += len(genres)
            event_director_sum += len(directors)
            event_actor_sum += len(actors)
            event_country_sum += len(countries)
            event_list.append(event)
    print(event_tag_sum / event_num)
    print(event_genre_sum/ event_num)
    print(event_director_sum / event_num)
    print(event_actor_sum / event_num)
    print(event_country_sum / event_num)
    print(event_tag_sum)
    print(event_genre_sum)
    print(event_director_sum)
    print(event_actor_sum)
    print(event_country_sum)
    print(event_num)
    event_sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
    with open(event_sorted_root, 'w') as event_sorted:
        for event in event_sorted_list:
            event_sorted.write(str(event['timestamp']) + '\t' + str(event['user']) + '\t' + str(event['movie']) + '\t'
                               + str(event['tag']) + '\t' + str(event['genre']) + '\t'
                               + str(event['director']) + '\t'
                               + str(event['actor']) + '\t'
                               + str(event['country']) + '\t'
                               + '\n')
    with open(indexer_root, 'wb') as indexer_data:
        pkl.dump(indexer, indexer_data)

#             # if user not in user_dict.keys():
#             #     user_dict[user] = len(user_dict.keys())
#             # if movie not in movie_dict.keys():
#             #     movie_dict[movie] = len(movie_dict.keys())
#             # if director not in director_dict.keys():
#             #     director_dict[director] = len(director_dict.keys())
#             # if country not in country_dict.keys():
#             #     country_dict[country] = len(country_dict.keys())
#             changed_eventgenres = []
#             for genre in event_genres:
#                 if genre not in genre_dict.keys():
#                     genre_dict[genre] = len(genre_dict.keys())
#                     changed_eventgenres.append(str(genre_dict[genre]))
#                 else:
#                     changed_eventgenres.append(str(genre_dict[genre]))
#             changed_eventactors = []
#             for actor in event_actors:
#                 if actor not in actor_dict:
#                     actor_dict[actor] = len(actor_dict.keys())
#                     changed_eventactors.append((str(actor_dict[actor])))
#                 else:
#                     changed_eventactors.append((str(actor_dict[actor])))
#             changed_eventtags = []
#             for tag in event_tags:
#                 if tag not in tag_dict:
#                     tag_dict[tag] = len(tag_dict.keys())
#                     changed_eventtags.append(str(tag_dict[tag]))
#                 else:
#                     changed_eventtags.append(str(tag_dict[tag]))
#
#     with open(event_sorted_root, 'w') as changeid_out:
#                 changeid_out.write(str(user_dict[user])+'\t'+str(movie_dict[movie])+'\t'
#                                     +';'.join(changed_eventgenres)+'\t'
#                                     +str(director_dict[director])+'\t'
#                                     +';'.join(changed_eventactors)+'\t'
#                                     +str(country_dict[country])+'\t'
#                                     +';'.join(changed_eventtags)+'\t'+str(timestamp)+'\n')
#     print(len(user_dict.keys()))
#     print(len(movie_dict.keys()))
#     print(len(genre_dict.keys()))
#     print(len(director_dict.keys()))
#     print(len(actor_dict.keys()))
#     print(len(country_dict.keys()))
#     print(len(tag_dict.keys()))
#     with open(user_id_root, 'wb') as user_id:
#         pkl.dump(user_dict, user_id)
#     with open(movie_id_root, 'wb') as movie_id:
#         pkl.dump(movie_dict, movie_id)
#     with open(genre_id_root, 'wb') as genre_id:
#         pkl.dump(genre_dict, genre_id)
#     with open(director_id_root, 'wb') as director_id:
#         pkl.dump(director_dict, director_id)
#     with open(actor_id_root, 'wb') as actor_id:
#         pkl.dump(actor_dict, actor_id)
#     with open(country_id_root, 'wb') as country_id:
#         pkl.dump(country_dict, country_id)
#     with open(tag_id_root, 'wb') as tag_id:
#         pkl.dump(tag_dict, tag_id)
#
# def sort_event():
#     event_list = []
#     with open(event_changed_root, 'r') as changed_event:
#         with open(event_sorted_root, 'w') as event_sorted:
#             for line in changed_event:
#                 event = {}
#                 value = line.strip().split('\t')
#                 event['user'] = value[0]
#                 event['movie'] = value[1]
#                 event['genre'] = value[2]
#                 event['director'] = value[3]
#                 event['actor'] = value[4]
#                 event['country'] = value[5]
#                 event['tag'] = value[6]
#                 event['timestamp'] = value[7]
#                 event_list.append(event)
#             sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
#             for event in sorted_list:
#                 event_sorted.write(str(event['timestamp']) + '\t' + str(event['user']) + '\t'
#                                     + str(event['movie']) + '\t'
#                                     + str(event['genre']) + '\t'
#                                     + str(event['director']) + '\t'
#                                     + str(event['actor']) + '\t'
#                                     + str(event['country']) + '\t'
#                                     + str(event['tag']) + '\n')
if __name__ == '__main__':
    user_list, movie_list, tag_list = choose_taggedmovies()
    concat_taggedmovies(user_list, movie_list, tag_list)
    user_list, movie_list, tag_list, genre_list, director_list, actor_list, country_list = concat_extrainfo()
    print(len(user_list))
    print(len(movie_list))
    print(len(tag_list))
    print(len(genre_list))
    print(len(director_list))
    print(len(actor_list))
    print(len(country_list))
    #1234 4417 3992 19 1947 20578 53
    change_taggedmovies(user_list, movie_list, tag_list, genre_list, director_list, actor_list, country_list)
    # sort_event()
