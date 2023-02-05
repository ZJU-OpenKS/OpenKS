import gensim
import string
import time
import re
# import spacy
from nltk.corpus import stopwords, wordnet
from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases

n_topics = 20

# event_review_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review.selected'
event_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review.selected'
event_data_root_withtopic = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_nv_withtopic.txt'
# event_data_root_withtopic_init = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_withtopic.txt'
event_data_topic_dict = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_nv_lda.dict'
event_data_topic_dict_nf = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_nv_lda_nf.dict'
# event_data_topic_dict_init = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review_lda.dict'
event_data_topic_corpus = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_nv_lda.corpus'
# event_data_topic_corpus_init = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review_lda.corpus'
event_data_topic_model = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_nv_lda.model'
# event_data_topic_model_init = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review_lda_20.model'

# def get_lemma(word):
#     lemma = wordnet.morthy(word)
#     if lemma is None:
#         return word
#     else:
#         return lemma
def get_filtered_tokens(review_text):
    # p_stemmer = EnglishStemmer()
    tokenizer = RegexpTokenizer(r'\w+')
    for c in string.punctuation:
        review_text = review_text.replace(c, '')
    tokens = tokenizer.tokenize(review_text)
    filtered = [w for w in tokens if w not in stopwords.words('english')]
    # filtered_lemma = [get_lemma(w) for w in filtered]
    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]
    return filtered

def prepare_text_for_lda():
    clean_up_review_list = []
    # p_stemmer = PorterStemmer()
    # nlp = spacy.load('en')
    index = 1
    with open(event_data_root, 'r') as event_data:
        for line in event_data:
            value = line.split('\t')
            review_text = value[5]
            review_text = re.sub(r'[^a-zA-Z]', ' ', review_text)
            review = review_text.lower()
            filtered = get_filtered_tokens(review)
            clean_up_review_list.append(filtered)
            if index % 1000 == 0:
                print(index)
            index += 1
    bigrams = Phrases(clean_up_review_list)
    dictionary = gensim.corpora.Dictionary(bigrams[clean_up_review_list])
    clean_up_review_list = bigrams[clean_up_review_list]
    # dictionary = gensim.corpora.Dictionary(clean_up_review_list)
    dictionary.save(event_data_topic_dict_nf)
    # dictionary.filter_extremes(keep_n=20000)
    dictionary.filter_extremes(no_below = 5, no_above = 0.2)
    dictionary.save(event_data_topic_dict)
    corpus = [dictionary.doc2bow(text) for text in clean_up_review_list]
    # corpus_tfidf = gensim.models.TfidfModel(corpus)[corpus]
    gensim.corpora.MmCorpus.serialize(event_data_topic_corpus, corpus)
    return dictionary, corpus

def load_model_for_lda():
    lda = gensim.models.LdaModel.load(event_data_topic_model)
    return lda

def runLda(saved_model):
    # ldamodel = gensim.models.LdaModel(corpus, num_topics=n_topics, id2word=dictionary, passes=200)
    if saved_model:
        ldamodel = load_model_for_lda()
        dictionary = gensim.corpora.Dictionary.load(event_data_topic_dict)
        corpus = gensim.corpora.MmCorpus(event_data_topic_corpus)
    else:
        # dictionary, corpus = prepare_text_for_lda()
        dictionary = gensim.corpora.Dictionary.load(event_data_topic_dict)
        dictionary.filter_extremes(no_below=5, no_above=0.2)
        corpus = gensim.corpora.MmCorpus(event_data_topic_corpus)
        ldamodel = gensim.models.LdaMulticore(corpus, num_topics=n_topics, id2word=dictionary, passes=30, workers=20)
        ldamodel.save(event_data_topic_model)

    review_index = 0
    with open(event_data_root_withtopic, 'w') as event_data_withtopic:
        with open(event_data_root, 'r') as event_data:
            for line in event_data:
                value = line.replace('\n', '')
                topic_prob = ldamodel.get_document_topics(corpus[review_index], minimum_probability=0.05)
                topiclist = []
                for topic in topic_prob:
                    topiclist.append(topic[0])
                review_index += 1
                value = value + '\t' + ';'.join(str(e) for e in topiclist) + '\n'
                event_data_withtopic.write(value)

def InferenceLda():
    ldamodel = load_model_for_lda()

    for topic in ldamodel.show_topics(num_topics=50):
        print(topic)
    # time.sleep(100000)

    corpus = gensim.corpora.MmCorpus(event_data_topic_corpus)
    dictionary = gensim.corpora.Dictionary.load(event_data_topic_dict)
    review_index = 0
    # with open(event_data_root_withtopic, 'w') as event_data_withtopic:
    #     with open(event_data_root, 'r') as event_data:
    #         for line in event_data:
    #             value = line.replace('\n', '')
    #             topic_prob = ldamodel.get_document_topics(corpus[review_index], minimum_probability=0.05)
    #             topiclist = []
    #             for topic in topic_prob:
    #                 topiclist.append(topic[0])
    #             review_index += 1
    #             value = value + '\t' + ';'.join(str(e) for e in topiclist) + '\n'
    #             event_data_withtopic.write(value)
    #             if review_index % 1000 == 0:
    #                 print(review_index)

    with open(event_data_root, 'r') as event_data:
        for line in event_data:
            value = line.replace('\n', '')
            topic_prob = ldamodel.get_document_topics(corpus[review_index], minimum_probability=0.05)
            for i in corpus[review_index]:
                print(i)
                print(dictionary.get(i[0]))
            print('_______________')
            print(topic_prob)
            time.sleep(5)
            # topiclist = []
            # for topic in topic_prob:
            #     topiclist.append(topic[0])
            review_index += 1



if __name__ == '__main__':
    # nltk.download('stopwords')
    # saved_model = False
    # runLda(saved_model)
    InferenceLda()
