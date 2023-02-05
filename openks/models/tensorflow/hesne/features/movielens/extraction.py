import logging
import os
from datetime import datetime

import pandas as pd

from features import Indexer, timestamp_delta_generator

rating_threshold = 4
actor_threshold = 3

def generate_indexer(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds, movie_director_ds, movie_genre_ds, movie_countries_ds):
    logging.info('generating indexer ...')
    min_time = 1e30
    max_time = -1

    indexer = Indexer(['user', 'tag', 'movie', 'actor', 'director', 'genre', 'country'])
    for line in user_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating_timestamp = float(line_items[3]) / 1000
        min_time = min(min_time, rating_timestamp)
        max_time = max(max_time, rating_timestamp)
        rateing = float(line_items[2])
        indexer.index('user', line_items[0])
        indexer.index('movie', line_items[1])

    for line in user_tags_movies_ds[1:]:
        line_items = line.split('\t')
        tag_timestamp = float(line_items[3]) / 1000
        min_time = min(min_time, tag_timestamp)
        max_time = max(max_time, tag_timestamp)
        indexer.index('user', line_items[0])
        indexer.index('movie', line_items[1])
        indexer.index('tag', line_items[2])

    for line in movie_actor_ds[1:]:
        line_items = line.split('\t')
        ranking = int(line_items[3])
        if ranking < actor_threshold and line_items[0] in indexer.mapping['movie']:
            indexer.index('actor', line_items[1])

    for line in movie_director_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            indexer.index('director', line_items[1])

    for line in movie_genre_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            indexer.index('genre', line_items[1])

    for line in movie_countries_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            indexer.index('country', line_items[1])

    with open('data/metadata.txt', 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Users: %d\n' % indexer.indices['user'])
        output.write('#Tags: %d\n' % indexer.indices['tag'])
        output.write('#Movies: %d\n' % indexer.indices['movie'])
        output.write('#Actors: %d\n' % indexer.indices['actor'])
        output.write('#Director: %d\n' % indexer.indices['director'])
        output.write('#Genre: %d\n' % indexer.indices['genre'])
        output.write('#Countriy: %d\n' % indexer.indices['country'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Rate: %d\n' % len(user_rates_movies_ds))
        output.write('#Attach: %d\n' % len(user_tags_movies_ds))
        output.write('#Played_by: %d\n' % len(movie_actor_ds))
        output.write('#Directed_by : %d\n' % len(movie_director_ds))
        output.write('#Has: %d\n' % len(movie_genre_ds))
        output.write('#Produced_in: %d\n' % len(movie_countries_ds))
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % datetime.fromtimestamp(min_time))
        output.write('To: %s\n' % datetime.fromtimestamp(max_time))

    return indexer

def parse_dataset(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds, movie_director_ds, movie_genre_ds, movie_countries_ds, indexer):
    logging.info('parsing dataset ...')
    rate = []
    attach = []
    played_by = []
    directed_by = []
    has = []
    produced_in = []

    for line in user_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating = float(line_items[2])
        rating_timestamp = float(line_items[3]) / 1000





def run(valid_delta, test_delta, observation_window):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_path = os.getcwd()
    os.chdir(dir_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    # user_rates_movies = pd.read_table('data/movielens/hetrec2011/user_ratedmovies-timestamps.dat', sep='\t', usercols=['userID', 'movieID', 'rating', 'timestamp'])
    user_tags_movies = pd.read_table('data/movielens/hetrec2011/user_taggedmovies-timestamps.dat', sep='\t', usercols=['userID', 'movieID', 'tagID', 'timestamp'])
    movie_actors = pd.read_table('data/movielens/hetrec2011/movie_actors.dat', sep='\t', usercols=['movieID', 'actorID'])
    movie_directors = pd.read_table('data/movielens/hetrec2011/movie_directors.dat', sep='\t', usercols=['movieID', 'directorID'])
    movie_genres = pd.read_table('data/movielens/hetrec2011/movie_genres.dat', sep='\t', usercols=['movieID', 'actorName'])
    moive_countries = pd.read_table('data/movielens/hetrec2011/movie_countries.dat', sep='\t', usercols=['movieID', 'actorName'])



    # with open('data/movielens/hetrec2011/user_ratedmovies-timestamps.dat') as user_rates_movies_ds:
    #     user_rates_movies_ds = user_rates_movies_ds.read().splitlines()
    # with open('data/movielens/hetrec2011/user_taggedmovies-timestamps.dat') as user_tags_movies_ds:
    #     user_tags_movies_ds = user_tags_movies_ds.read().splitlines()
    # with open('data/movielens/hetrec2011/movie_actors.dat', encoding='latin-1') as movie_actor_ds:
    #     movie_actor_ds = movie_actor_ds.read().splitlines()
    # with open('data/movielens/hetrec2011/movie_directors.dat', encoding='latin-1') as movie_director_ds:
    #     movie_director_ds = movie_director_ds.read().splitlines()
    # with open('data/movielens/hetrec2011/movie_genres.dat') as movie_genre_ds:
    #     movie_genre_ds = movie_genre_ds.read().splitlines()
    # with open('data/movielens/hetrec2011/movie_countries.dat') as movie_countries_ds:
    #     movie_countries_ds = movie_countries_ds.read().splitlines()


    valid_delta = timestamp_delta_generator(years=valid_delta)
    test_delta = timestamp_delta_generator(years=test_delta)
    test_end = datetime(2009, 1, 1).timestamp()
    valid_end = test_end - test_delta
    train_end = valid_end - valid_delta




