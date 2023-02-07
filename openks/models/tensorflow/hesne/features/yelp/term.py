from sklearn.feature_extraction.text import TfidfVectorizer

review_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review.collection'
term_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/phrase_tf_idf_top10.txt'

top_K = 5
corpus = []
corpus_review_id = []
# count = 10000

def generate_term():
    count = 0
    with open(review_out_root, 'r') as review_data:
        for line in review_data:
            value = line.split('\t')
            corpus.append(value[3])
            corpus_review_id.append(value[0])
            count += 1
            # count -= 1
            # if count == 0:
            #     pass
    print('Finished reading:', str(count))

    def tokenize(text):
        return text.split(' ')

    vectorizer = TfidfVectorizer(tokenizer=tokenize, min_df=5, max_df=0.03, stop_words='english')
    m = vectorizer.fit_transform(corpus)
    print('Finished tf-idf.', m.shape[0], m.shape[1])

    print(vectorizer.get_stop_words())
    phrases = vectorizer.get_feature_names()
    with open(term_out_root, 'w') as output:
        for i in range(m.shape[0]):
            d = m.getrow(i)
            s = zip(d.indices, d.data)
            sorted_s = sorted(s, key=lambda v: v[1], reverse=True)
            indices = [element[0] for element in sorted_s]
            output.write(corpus_review_id[i] + '\t')
            for i in range(min(top_K, len(indices))):
                output.write(phrases[indices[i]])
                output.write('\t')
            output.write('\n')

if __name__ == '__main__':
    generate_term()
