# Copyright (c) 2021 OpenKS Authors, DCD Research Lab, Zhejiang University. 
# All Rights Reserved.

import re
import os
import yaml
import logging
import jieba
import jieba.posseg as pg
import json
from gensim.models import Word2Vec

from ...model import MLModel
from .topic_similarity_rank import SimilarityRank

DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)

@MLModel.register("keyphrase-rake", "MLLib")
class Rake(MLModel):
    ''' Key phrase extration with Rake modified for Chinese text.
    Reference project: https://github.com/fabianvf/python-rake
    Reference paper: Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). Automatic Keyword Extraction from Individual Documents
    '''
    def __init__(self, args):
        if not args['stopword']:
            logging.warning("No file for stop words is configured!")
            raise Exception("请先指定中文停用词文件路径！")
        else:
            sw_file = os.path.join(DIR, args['stopword'])
        sw_open = args['stopword_open']
        self.stop_words = self.load_stop_words(sw_file)
        self.stop_words_open = self.load_stop_words(sw_open)
        self.params = args['params']
        # jieba.load_userdict(sw_file)
    
    def is_duplicate(self, new_word, words):
        for word in words:
            if new_word in word:
                return True 
        return False 

    def process(self, dataset, top_k=100):
        texts = dataset[1]
        result = []
        for text in texts:
            sentences = self.split_sentences(text)
            phrases = self.generate_candidate_keywords(sentences)
            if self.params['SUFFIX_REMOVE']:
                phrases = self.delete_suffix(phrases)
            word_scores = self.calculate_word_scores(phrases)
            keyword_candidates = self.generate_candidate_keyword_scores(phrases, word_scores)
            sorted_keywords = sorted(keyword_candidates.items(), key=lambda x:x[1], reverse=True)
            result.append([k for k in sorted_keywords[:top_k] if k[1] >= self.params['MIN_SCORE']])
        return result


    def load_stop_words(self, sw_file):
        stop_words = []
        for line in open(sw_file):
            line = line.strip()
            if line.strip()[0:1] != "#":
                stop_words.append(line)
        return stop_words
    

    def split_sentences(self, text):
        sentence_delimiters = re.compile(u'[。\.、：:；;,，"（）,“”？//><]')
        sentences = sentence_delimiters.split(text)
        return sentences
    
    
    def is_number(self, s):
        try:
            float(s) if '.' in s else int(s)
            return True
        except ValueError:
            return False


    def generate_candidate_keywords(self, sentence_list):
        phrase_list = []
        for sentence in sentence_list:
            s = sentence
            # assumption：单字且被分词的为真正的停用词，否则不被替换
            if self.params['STOPWORD_SINGLE_CHECK']:
                s = ' '.join(jieba.cut(sentence))
                for stop in self.stop_words:
                    if len(stop) == 1:
                        if " " + stop + " " in s:
                            s = re.sub(stop,"|",s)
                        elif len(s) > 0 and stop == s[0] and stop+" " in s:
                            s = re.sub(stop,"|",s)
                        else:
                            continue
                s = ''.join(s.split(' '))

            # 处理多字停用词，不切词直接替换
            for stop in self.stop_words:
                if len(stop) > 1:
                    s = re.sub(stop, '|', s)
            # print(s)

            # 使用开放停用词表，对多字停用词进行召回
            if self.params['OPEN_STOPWORD']:
                for stop in self.stop_words_open:
                    if len(stop) > 1:
                        s = s.replace(stop, '|')

            phrases = s.split("|")

            for phrase in phrases:
                phrase = phrase.strip().lower().replace(" ","")
                # phrase = phrase.rstrip("或").lstrip("其").lstrip("以").rstrip("高").lstrip("除").rstrip("时").rstrip("有").lstrip("且").lstrip("经").rstrip("及")
                if phrase != "" and len(phrase) >= self.params['MIN_WORD_LEN']:
                    if phrase not in self.stop_words:
                        phrase_list.append(phrase)
        # print(phrase_list)
                        
        return phrase_list
    
    
    def separate_words(self, text):
        """
        Utility function to return a list of all words that are have a length greater than a specified number of characters.
        @param text The text that must be split in to words.
        @param min_word_return_size The minimum no of characters a word must have to be included.
        """
        # separate words for Chinese text
        if self.params['WORD_SEPARATOR']:
            text = jieba.lcut(text)

        words = []
        for single_word in [w for w in text]:
            current_word = single_word.strip().lower()
            # leave numbers in phrase, but don't count as words, since they tend to invalidate scores of their phrases
            if current_word != '' and not self.is_number(current_word):
                words.append(current_word)
        # print(words)
        return words
    
    
    def calculate_word_scores(self, phraseList):
        word_frequency = {}
        word_degree = {}
        for phrase in phraseList:
            word_list = self.separate_words(phrase)
            word_list_length = len(word_list)
            word_list_degree = word_list_length - 1
            for word in word_list:
                word_frequency.setdefault(word, 0)
                word_frequency[word] += 1
                word_degree.setdefault(word, 0)
                word_degree[word] += word_list_degree

        for item in word_frequency:
            word_degree[item] = word_degree[item] + word_frequency[item]

        # Calculate Word scores = deg(w)/frew(w)
        word_score = {}
        for item in word_frequency:
            word_score.setdefault(item, 0)
            word_score[item] = word_degree[item] / (word_frequency[item] * 1.0)
        return word_score
    
    
    def generate_candidate_keyword_scores(self, phrase_list, word_score, minFrequency=1):
        keyword_candidates = {}
        for phrase in phrase_list:
            if "".join(phrase_list).count(phrase) >= minFrequency:
                keyword_candidates.setdefault(phrase, 0)
                word_list = self.separate_words(phrase)
                candidate_score = 0
                for word in word_list:
                    candidate_score += word_score[word]
                keyword_candidates[phrase] = candidate_score
        return self.normalize_scores(keyword_candidates)



    def judge_suffix(self, word):
        suffixes = ['器','器件','技术','结构','装置','方法','机','生产线','电路','单元','计','芯片','工艺','系统','模块','组合物']
        for suffix in suffixes:
            suffix_length = len(suffix)
            if len(word) < suffix_length:
                continue
            if word[len(word) - suffix_length:] == suffix:
                return True    
        return False

    def delete_suffix(self, phrases):
        phrases_suffix_removed = []
        # 基于末尾词性假设，删除非正常末尾词
        for p in phrases:
            word_list = list(pg.cut(p))
            word = [x for x, y in word_list]
            pos = [y for x, y in word_list]
            for k in range(len(pos) - 1, -1, -1):
                if pos[k] in ['n', 'v', 'nz', 'ng', 'vn', 'l', 'q', 'zg', 'eng', 'p', 'd', 'ns', 'b']:
                    phrases_suffix_removed.append(''.join(word[:k+1]))
                    break

        # 基于长短语末尾词模式假设，删除非正常末尾词
        final_result = []
        for p in phrases_suffix_removed:
            length = len(p)
            if length >= self.params['MIN_WORD_LEN'] and (length < 18 or (length >= 18 and self.judge_suffix(p) == True)):
                final_result.append(p)
        # print(final_result)
        return final_result


    def normalize_scores(self, keyword_scores):
        keyword_candidates_normalized = {}
        mincols = 1000000.0
        maxcols = 0.0
        for k,v in keyword_scores.items():
            mincols = min(mincols, float(v))
            maxcols = max(maxcols, float(v))

        maxcols = maxcols + 1
       
        for k,v in keyword_scores.items():
            keyword_candidates_normalized[k] = (v - mincols) / (maxcols - mincols)
        return keyword_candidates_normalized


@MLModel.register("keyphrase-rake-topic", "MLLib")
class TopicRake(MLModel):
    ''' A topic relevant key phrase extraction on top of Rake.
    Using a combined score between phrase importance and semantic similarity.
    Reference paper: Technical Phrase Extraction for Patent Mining: A Multi-level Approach
    '''
    def __init__(self, args):
        self.rake = Rake(args)
        self.similarRank = SimilarityRank(args)
        self.rank_alg = args['rank']
        self.params = args['params']

    def process(self, dataset, top_k=100):
        topic_text = dataset
        key_phrases = self.rake.process(dataset)
        total_result = []
        total_count = len(topic_text[0])
        for i in range(total_count):
            result_dict = {}
            topic = topic_text[0][i]
            key_phrase = key_phrases[i]
            phrases = [item[0] for item in key_phrase]
            topic_phrases = self.similarRank.rank(topic, phrases)
            for j in range(len(phrases)):
                if self.rank_alg == 'average':
                    item = phrases[j]
                    for k in range(len(topic_phrases)):
                        if topic_phrases[k][0] == item:
                            result_dict[item] = (topic_phrases[k][1] + key_phrase[j][1]) / 2
                        else:
                            continue
            sorted_list = []
            sorted_keys = sorted(result_dict, key=result_dict.get, reverse=True)
            
            for w in result_dict:
                if result_dict[w] < self.params['SIM_SCORE']:
                    continue
                sorted_list.append([w, result_dict[w]])
            total_result.append(sorted_list)
        return total_result


if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, Loader=yaml.Loader)
    
    if config['extractor'] == 'rake':
        extractor = Rake(config)
        res = extractor.process()
        with open(config['result_dir'] + '/' + config['extractor'] + '_', "w") as out:
            for res_item in res:
                out.write(json.dumps(res_item, ensure_ascii=False) + '\n')

    elif config['extractor'] == 'topic-rake':
        extractor = TopicRake(config)
        res = extractor.process()
        with open(config['result_dir'] + '/' + config['extractor'] + '_' + str(threshold), "w") as out:
            for res_item in res:
                out.write(json.dumps(res_item, ensure_ascii=False) + '\n')
    