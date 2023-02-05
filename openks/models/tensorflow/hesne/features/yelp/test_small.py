import time
import pickle as pkl

event_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_withtopic_tfidf.txt'
event_sorted_root_whole = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_whole.txt'
event_sorted_root_small = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small.txt'
event_sorted_root_small_changed = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small_changed.txt'
event_sorted_root_small_train = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small_train.txt'
event_sorted_root_small_valid = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small_valid.txt'
event_sorted_root_small_test = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event_sorted_small_test.txt'
user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/user_small_id.pkl'
business_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business_small_id.pkl'
term_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/term_small_id.pkl'
review_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review_small_id.pkl'

# copyline = 30000
def event_small():
    line_num = 0
    with open(event_sorted_root_whole, 'r') as event:
        with open(event_sorted_root_small, 'w') as event_small:
            for line in event:
                # if line_num<30000:
                event_small.write(line)
                # line_num += 1

def change_eventwithid():
    review_dict = {}
    user_dict = {}
    business_dict = {}
    term_dict = {}

    with open(event_sorted_root_small, 'r') as event_data:
        with open(event_sorted_root_small_changed, 'w') as changeid_out:
            for line in event_data:
                value = line.split('\t')
                review_id = value[0]
                star = value[1]
                event_timestamp = value[2]
                user_id = value[3]
                business_id = value[4]
                event_term_value = value[6].replace('\n', '')
                event_terms = event_term_value.split(';')

                if review_id not in review_dict.keys():
                    review_dict[review_id] = len(review_dict.keys())
                if user_id not in user_dict.keys():
                    user_dict[user_id] = len(user_dict.keys())
                if business_id not in business_dict.keys():
                    business_dict[business_id] = len(business_dict.keys())
                changed_eventterms = []
                for term in event_terms:
                    if term not in term_dict:
                        term_dict[term] = len(term_dict.keys())
                        changed_eventterms.append(str(term_dict[term]))
                    else:
                        changed_eventterms.append(str(term_dict[term]))
                changeid_out.write(str(review_dict[review_id]) + '\t' + str(star) + '\t' + str(event_timestamp) + '\t' + str(user_dict[user_id]) + '\t' + str(business_dict[business_id]) + '\t' + ';'.join(changed_eventterms) + '\n')

    with open(user_id_root, 'wb') as user_id:
        pkl.dump(user_dict, user_id)
    with open(business_id_root, 'wb') as business_id:
        pkl.dump(business_dict, business_id)
    with open(term_id_root, 'wb') as term_id:
        pkl.dump(term_dict, term_id)
    with open(review_id_root, 'wb') as review_id:
        pkl.dump(term_dict, review_id)

def split_event(train_p, valid_p, test_p):
    with open(event_sorted_root_small_changed, 'r') as event_sorted:
        num_lines = sum(1 for line in event_sorted)
    with open(event_sorted_root_small_changed, 'r') as event_sorted:
        with open(event_sorted_root_small_train, 'w') as event_train:
            with open(event_sorted_root_small_valid, 'w') as event_valid:
                with open(event_sorted_root_small_test, 'w') as event_test:
                    cur_num = 0
                    for line in event_sorted:
                        if cur_num < int(train_p*num_lines):
                            event_train.write(line)
                        elif int(train_p*num_lines) <= cur_num < int((train_p+valid_p)*num_lines):
                            event_valid.write(line)
                        else:
                            event_test.write(line)
                        cur_num += 1

def sort_event():
    event_list = []
    with open(event_root, 'r') as changed_event:
        with open(event_sorted_root_whole, 'w') as event_sorted:
            for line in changed_event:
                event = {}
                value = line.split('\t')
                event['review_id'] = value[0]
                event['user_id'] = value[1]
                event['business_id'] = value[2]
                event['star'] = value[3]
                event['text'] = value[4]
                event_time = value[5]
                event_timearray = time.strptime(event_time, '%Y-%m-%d')
                event_timestamp = time.mktime(event_timearray)
                event['timestamp'] = event_timestamp
                event['topics'] = value[7].replace('\n', '')
                event_list.append(event)
            sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
            for event in sorted_list:
                event_sorted.write(str(event['review_id']) + '\t' + str(event['star']) + '\t' +
                                   str(event['timestamp']) + '\t' + str(event['user_id']) + '\t' +
                                   str(event['business_id']) + '\t' + str(event['text']) + '\t' +
                                   str(event['topics']) + '\n')


if __name__ == '__main__':
    sort_event()
    event_small()
    change_eventwithid()
    split_event(0.8,0.1,0.1)

