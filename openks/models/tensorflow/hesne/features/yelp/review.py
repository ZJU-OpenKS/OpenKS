import json
import re
# from nltk.corpus import stopwords, wordnet
# from nltk.stem.porter import PorterStemmer
# from nltk.stem.snowball import EnglishStemmer
# from nltk.tokenize import RegexpTokenizer

review_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/review.json'
review_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review.selected'
user_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/user.dict'
business_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/business.dict'
user_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/user.selected'
business_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/business.selected'
#
def contain_zh(word):
    zh_pattern = re.compile(u'[\u4e00-\u9fff]+')
    match = zh_pattern.search(word)
    return match
#
# def clean_review(review):
#     letters_only = re.sub('[^a-zA-Z]', ' ', review)
#     return letters_only
# def get_filtered_tokens(review_text):
#     p_stemmer = EnglishStemmer()
#     tokenizer = RegexpTokenizer(r'\w+')
#     for c in string.punctuation:
#         review_text = review_text.replace(c, '')
#     tokens = tokenizer.tokenize(review_text)
#     filtered = [w for w in tokens if w not in stopwords.words('english')]
#     # filtered_lemma = [get_lemma(w) for w in filtered]
#     # ps = PorterStemmer()
#     filtered = [p_stemmer.stem(w) for w in filtered]
#     return filtered

def choose_review():
    users = dict()
    restaurants = dict()
    with open(user_out_root, 'r') as user_out:
        for line in user_out:
            value = line.split('\t')
            users[value[0]] = 0

    with open(business_out_root, 'r') as business_out:
        for line in business_out:
            value = line.split('\t')
            restaurants[value[0]] = 0

    with open(review_data_root, 'r') as review_data:
        count = 0
        for line in review_data:
            review = json.loads(line)
            if review['user_id'] not in users.keys():
                continue
            if review['business_id'] not in restaurants.keys():
               continue
            # review_letter = clean_review(review['text'])
            # if review_letter == '':
            #     print(review['text'])
            #     continue
            if contain_zh(review['text']):
                # print(review['text'])
                continue
            users[review['user_id']] += 1
            restaurants[review['business_id']] += 1
        while min(users.values())<10 or min(restaurants.values())<10:
            count += 1
            print('Loop select review times:', str(count))
            users = {key:0 for key, val in users.items() if val >= 10}
            restaurants = {key:0 for key, val in restaurants.items() if val>=10}
            with open(review_data_root, 'r') as review_data:
                for line in review_data:
                    review = json.loads(line)
                    if review['user_id'] not in users.keys():
                        continue
                    if review['business_id'] not in restaurants.keys():
                        continue
                    users[review['user_id']] += 1
                    restaurants[review['business_id']] += 1
            print(len(users))
            print(len(restaurants))

    with open(review_selected_root, 'w') as fout:
        with open(review_data_root, 'r') as review_data:
            for line in review_data:
                review = json.loads(line)
                if review['user_id'] not in users.keys():
                    continue
                if review['business_id'] not in restaurants.keys():
                    continue
                try:
                    fout.write(review['review_id'] + '\t' + review['date'] + '\t' + str(review['stars']) + '\t' + review['user_id'] + '\t' +
                    review['business_id'] + '\t' + review['text'].replace('\n', ' ').replace('\r', '').lower() + '\n')
                except:
                    print(review['review_id'])
                    print(review['user_id'])
                    print(review['business_id'])
                    print(review['stars'])
                    print(review['text'].replace('\n', ' ').replace('\r', '').lower())
                    print(review['date'])
    print('Review save done')

    count = 0
    with open(user_out_root, 'r') as user_out:
        with open(user_selected_root, 'w') as user_selected:
            for line in user_out:
                value = line.split('\t')
                if value[0] in users.keys():
                    user_selected.write(line + '\t' + str(count))
                    count += 1
    print('User selected done')

    count = 0
    with open(business_out_root, 'r') as business_out:
        with open(business_selected_root, 'w') as business_selected:
            for line in business_out:
                value = line.split('\t')
                if value[0] in restaurants.keys():
                    business_selected.write(line + '\t' + str(count))
                    count += 1
    print('Business selected done')

if __name__ == '__main__':
    choose_review()
