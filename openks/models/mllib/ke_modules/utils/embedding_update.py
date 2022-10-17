import jieba
from gensim.models import Word2Vec, KeyedVectors
import logging
import re

logging.basicConfig(level=logging.INFO)

def get_corpus(args):
    data_dir = args['data_dir']
    corpus = []
    sentence_delimiters = re.compile(u'[。\.、：:；;,，"（）,“”？//><@]')
    with open(data_dir, "r") as f:
        for line in f:
            sentences = sentence_delimiters.split(line)
            for s in sentences:
                corpus.append([item.lower() for item in jieba.lcut(s, cut_all=True)])
            # corpus.append(list(line.lower()))
    return corpus

def update_word_embedding(args):

    domain_data = get_corpus(args)

    w2v_model = Word2Vec(size=300, sg=1, min_count=1)
    w2v_model.build_vocab(domain_data)
    pretrained_model = KeyedVectors.load_word2vec_format(args['pretrained'], binary=False)

    w2v_model.build_vocab([list(pretrained_model.vocab.keys())], update=True)
    w2v_model.intersect_word2vec_format(args['pretrained'], binary=False, lockf=1.0)
    w2v_model.train(domain_data, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

    w2v_model.save(args['finetuned'])
    
    print("Complete word embedding training!")


if __name__ == "__main__":
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = f.read()
    config = yaml.load(config, Loader=yaml.Loader)
    update_word_embedding(config)
