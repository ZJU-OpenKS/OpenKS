import random
import pickle as pkl
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
embedding_outroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/embedding_small_out_nt.pkl'
# train_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/embedding_small_train_lr.txt'
train_lrroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/embedding_small_train_lr.txt'
test_lrroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/embedding_small_test_lr.txt'
event_sorted_root_train = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_traindata.txt'
# event_sorted_root_small_valid = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small_valid.txt'
event_sorted_root_test = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_testdata.txt'
user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/user_id.pkl'
business_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/business_id.pkl'
term_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/term_id.pkl'
review_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_id.pkl'

def get_negsample(sample_list):
    negsample_list = []
    with open(user_id_root, 'rb') as user_data:
        user_id = pkl.load(user_data)
    with open(business_id_root, 'rb') as business_data:
        business_id = pkl.load(business_data)
    with open(term_id_root, 'rb') as term_data:
        term_id = pkl.load(term_data)
    for event in sample_list:
        negevent = {}
        negevent['label'] = 0
        negevent['user_id'] = random.sample(list(user_id.values()), 1)[0]
        negevent['business_id'] = random.sample(list(business_id.values()), 1)[0]
        len_term = len(event['event_term_value'].split(';'))
        neg_term = random.sample(list(term_id.values()), len_term)
        neg_term = [str(term) for term in neg_term]
        negevent['event_term_value'] = ';'.join(neg_term)
        negsample_list.append(negevent)
    return negsample_list

def sample_traindata():
    event_list = []
    with open(event_sorted_root_train, 'r') as train_data:
        for line in train_data:
            value = line.split('\t')
            event = {}
            event['label'] = 1
            event['user_id'] = value[3]
            event['business_id'] = value[4]
            event['event_term_value'] = value[5].replace('\n', '')
            event_list.append(event)
    sample_num = int(0.1*len(event_list))
    sample_list = random.sample(event_list, sample_num)
    negsample_list = get_negsample(sample_list)
    train_list = sample_list + negsample_list
    random.shuffle(train_list)
    with open(train_lrroot, 'w') as train_lr:
        for event in train_list:
            train_lr.write(str(event['label'])+'\t'+str(event['user_id'])+'\t'+str(event['business_id'])+'\t'+event['event_term_value']+'\n')

def sample_testdata():
    event_list = []
    with open(event_sorted_root_test, 'r') as test_data:
        for line in test_data:
            value = line.split('\t')
            event = {}
            event['label'] = 1
            event['user_id'] = value[3]
            event['business_id'] = value[4]
            event['event_term_value'] = value[5].replace('\n', '')
            event_list.append(event)
    sample_list = event_list
    negsample_list = get_negsample(sample_list)
    test_list = sample_list + negsample_list
    random.shuffle(test_list)
    with open(test_lrroot, 'w') as test_lr:
        for event in test_list:
            test_lr.write(str(event['label']) + '\t' + str(event['user_id']) + '\t' + str(event['business_id']) + '\t' + event[
                'event_term_value'] + '\n')


def train_lr():
    event_embedding_list = []
    event_label_list = []
    with open(embedding_outroot, 'rb') as embedding_data:
        node_embedding = pkl.load(embedding_data)
    with open(train_lrroot, 'r') as train_data:
        for line in train_data:
            value = line.split('\t')
            review_label = int(value[0])
            user_id = int(value[1])
            business_id = int(value[2])
            event_term_value = value[3].replace('\n', '')
            event_terms = event_term_value.split(';')
            event_terms = [int(term) for term in event_terms]
            event_terms = np.array(event_terms)
            user_embedding = node_embedding[0][user_id]
            business_embedding = node_embedding[1][business_id]
            terms_embedding = node_embedding[2][event_terms]
            term_embedding = np.mean(terms_embedding, axis=0)
            # event_embedding = np.concatenate((user_embedding, business_embedding, term_embedding), axis=0)
            event_embedding = user_embedding + business_embedding + term_embedding
            event_embedding_list.append(event_embedding)
            event_label_list.append(review_label)
    event_embedding_list = np.array(event_embedding_list)
    event_label_list = np.array(event_label_list)
    logreg = LogisticRegression()
    logreg.fit(event_embedding_list, event_label_list)
    return logreg, node_embedding

def test_lr(logreg, node_embedding):
    event_embedding_list = []
    label_true_list = []
    r_label_list = []
    test = 1
    with open(test_lrroot, 'r') as test_data:
        for line in test_data:
            value = line.split('\t')
            review_label = int(value[0])
            user_id = int(value[1])
            business_id = int(value[2])
            event_term_value = value[3].replace('\n', '')
            event_terms = event_term_value.split(';')
            event_terms = [int(term) for term in event_terms]
            event_terms = np.array(event_terms)
            user_embedding = node_embedding[0][user_id]
            business_embedding = node_embedding[1][business_id]
            terms_embedding = node_embedding[2][event_terms]
            term_embedding = np.mean(terms_embedding, axis=0)
            # event_embedding = np.concatenate((user_embedding, business_embedding, term_embedding), axis=0)
            event_embedding = user_embedding + business_embedding + term_embedding
            event_embedding_list.append(event_embedding)
            label_true_list.append(review_label)
            # test += 1
            r = np.tanh(np.sum(np.square(event_embedding))/2)
            r_label_list.append(r)
            # print(r)
            # print(event_embedding)
            # print(np.sum(np.square(event_embedding))/2)
            # if test == 50:
            #     break

        # pre_label = logreg.predict(event_embedding)
    event_embedding_list = np.array(event_embedding_list)
    pre_label_list = logreg.predict_proba(event_embedding_list)
    pre_label_list = pre_label_list[:, 1]
    label_true_list = np.array(label_true_list)
    # print(pre_label_list)
    # print(label_true_list)
    # print(r_label_list)
    # print(logreg.predict(event_embedding_list))
    # rmse = np.sqrt(mean_squared_error(label_true_list, pre_label_list))
    mae = mean_absolute_error(label_true_list, r_label_list)
    rmse = np.sqrt(mean_squared_error(label_true_list, r_label_list))
    print('mae:'+str(mae))
    print('rmse:'+str(rmse))


if __name__ == '__main__':
    # sample_traindata()
    # sample_testdata()
    logreg, node_embedding = train_lr()
    test_lr(logreg, node_embedding)
