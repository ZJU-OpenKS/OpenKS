import numpy as np
import gensim
import jieba
import yaml
import logging

from gensim.models import Word2Vec
import gensim.models.keyedvectors as word2vec

logging.basicConfig(level=logging.INFO)

class SimilarityRank(object):

    def __init__(self, config):
        stop_word_path = config['stopword']
        self.stop_words = self.load_stop_words(stop_word_path)
        # self.word_embedding = word2vec.KeyedVectors.load_word2vec_format(embedding_path)
        self.word_embedding = Word2Vec.load(config['finetuned'])

    def load_stop_words(self, sw_file):
        stop_words = []
        for line in open(sw_file):
            line = line.strip()
            if line.strip()[0:1] != "#":
                stop_words.append(line)
        return stop_words

    def get_topic_embedding(self, topics):
        topic_word_embedding_list = []
        if len(topics) == 0:
            topic_word_embedding_list.append(np.zeros(300))
        for phrase in topics:
            if not phrase:
                topic_word_embedding_list.append(np.zeros(300))
            else:
                for word in jieba.lcut(phrase, cut_all=True):
                    if word not in self.word_embedding:
                        topic_word_embedding_list.append(np.zeros(300))
                    else:
                        topic_word_embedding_list.append(self.word_embedding[word])

        topic_embedding = np.array(np.mean(topic_word_embedding_list, axis=0))
        return topic_embedding

    def get_phrases_embeddings(self, phrases):
        phrase_embedding_dict = {}
        for phrase in phrases:
            phrase_word_embedding_list = []
            for word in jieba.lcut(phrase, cut_all=True):
                if word not in self.word_embedding:
                    phrase_word_embedding_list.append(np.zeros(300))
                else:
                    phrase_word_embedding_list.append(self.word_embedding[word])
            phrase_embedding = np.array(np.mean(phrase_word_embedding_list, axis=0))
            phrase_embedding_dict[phrase] = phrase_embedding
        return phrase_embedding_dict

    def rank(self, topics, phrases, algorithm='cosine'):
        topic_embedding = self.get_topic_embedding(topics)
        phrase_embedding_dict = self.get_phrases_embeddings(phrases)
        similarity_dict = {}
        for phrase, embedding in phrase_embedding_dict.items():
            if algorithm == 'cosine':
                norm = np.linalg.norm(embedding) * np.linalg.norm(topic_embedding)
                if norm == 0.0:
                    similarity = 0.5
                else:
                    similarity = np.dot(embedding, topic_embedding) / norm
                    similarity = similarity.item()
            similarity_dict[phrase] = similarity

        sorted_list = []
        sorted_keys = sorted(similarity_dict, key=similarity_dict.get, reverse=True)
        for w in sorted_keys:
            sorted_list.append([w, similarity_dict[w]])
        return sorted_list


if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, Loader=yaml.Loader)
    sim_rank = SimilarityRank(config)
    rank_res = sim_rank.rank(["电子级酸性试剂"], ['14纳米及以下宽先进制程电子级盐酸', '抗干扰性痕量杂质离子检测', '湿电子化学材料纯化除杂', '集成电路芯片先进制程', '国际先进水平', '除杂工艺选型', '电子级盐酸', '重大科学问题', '酸性试剂产品', '电子级硝酸', '电子级硫酸', '国产化配套', '分析检测', '金属离子', '国际smei标准', '1um颗粒数量', '04um颗粒数量', '12英寸晶圆'])
    print(rank_res)
