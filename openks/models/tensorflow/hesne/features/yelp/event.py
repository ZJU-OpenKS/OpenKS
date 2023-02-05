# import json
# import re
# import time
#
# business_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business.selected'
# user_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/user.selected'
# review_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review.collection'
# term_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/term_tf_idf_top10.txt'
# user_id_old = dict()
# business_id_old = dict()
# term_dict = dict()
#
# def generate_event():
#     with open(user_selected_root, 'r') as user_selected:
#         for line in user_selected:
#             value = line.split('\t')
#             user_id = value[-1]
#             user_id_old[value[0]] = user_id
#
#     with open(business_selected_root, 'r') as business_selected:
#         for line in business_selected:
#             value = line.split('\t')
#             business_id = value[-1]
#             # name = re.sub('[^a-z\-\']+', ' ', value[3].lower())
#             # busi_name[business] = '#'.join(name.split())
#             business_id_old[value[0]] = business_id
#     print('Load user and business done')
#
#
#     count = 0
#     with open(review_out_root, 'r') as review_out:
#         with open(term_out_root, 'w') as term_out:
#             for line in review_out:
#                 rid = ids_dict[count]
#                 count += 1
#                 value = line.split()
#                 if len(value) == 0:
#                     print(count+1)
#                     continue
#                 phrase_dict[rid] = '#'.join(value)
#
#             print('Load phrases', len(phrase_dict))
#
#         # load review
#         data = open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/review.json')
#         fout = open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event.txt.all.city', 'w')
#         review_list = []
#         review_time = []
#         for line in data:
#             jdata = json.load(line)
#             busi = jdata['business_id']
#             user = jdata['user_id']
#             rid = jdata['review_id']
#             rtime = jdata['date']
#
#             if busi not in busi_name:
#                 continue
#
#             if rid not in phrase_dict:
#                 continue
#
#             review_list.append('%s\t%s\t%s\t%s\t%s\t%s\t\n' % (rid, user, busi, busi_name[busi], busi_zip[busi], phrase_dict[rid]))
#             review_time.append(rtime)
#
#         review_zip = zip(review_list, review_time)
#         review_time = sorted(review_zip, key=lambda v:time.strptime(v[1], '%Y-%m-%d')[0:3])
#
#         for review in review_zip:
#             fout.write('%s\t%s\t\n' % (review[1], review[2]))
#
#         fout.close()

import pickle as pkl
import time
import re

event_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_nv_withtopic.txt'
event_changeout_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_out.txt'
event_sorted_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_sorted.txt'
# event_sorted_trainroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_train.txt'
# event_sorted_validroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_valid.txt'
# event_sorted_testroot = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/event_test.txt'
user_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/user_id.pkl'
business_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/business_id.pkl'
term_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/term_id.pkl'
review_id_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/review_id.pkl'
user_dict = {}
business_dict = {}
term_dict = {}
review_dict = {}
review_filted_list = []
def change_eventwithid():
#     with open(review_filted, 'r') as event_data:
#         for line in event_data:
#             value = line.split('\t')
#             review_id = value[0]
#             review_filted_list.append(review_id)

    with open(event_data_root, 'r') as event_data:
        with open(event_changeout_root, 'w') as changeid_out:
            for line in event_data:
                value = line.split('\t')
                review_id = value[0]
                user_id = value[3]
                business_id = value[4]
                star = value[2]
                event_time = value[1]
                event_timearray = time.strptime(event_time, '%Y-%m-%d')
                event_timestamp = time.mktime(event_timearray)
                event_term_value = value[6].replace('\n', '')
                event_terms = event_term_value.split(';')
                if len(event_terms) > 10:
                    continue
                # if review_id not in review_filted_list:
                #     continue
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

    # with open(user_id_root, 'w') as user_id:
    #     for user, id in user_dict.items():
    #         user_id.write(str(user) + '\t' + str(id) + '\n')
    # with open(business_id_root, 'w') as business_id:
    #     for business, id in business_dict.items():
    #         business_id.write(str(business) + '\t' + str(id) + '\n')
    # with open(term_id_root, 'w') as term_id:
    #     for term, id in term_dict.items():
    #         term_id.write(str(term) + '\t' + str(id) + '\n')
    # with open(review_id_root, 'w') as review_id:
    #     for review, id in review_dict.items():
    #         review_id.write(str(review) + '\t' + str(id) + '\n')



def sort_event():
    event_list = []
    with open(event_changeout_root, 'r') as changed_event:
        with open(event_sorted_root, 'w') as event_sorted:
            for line in changed_event:
                event = {}
                value = line.split('\t')
                event['review'] = value[0]
                event['star'] = value[1]
                event['timestamp'] = value[2]
                event['user'] = value[3]
                event['business'] = value[4]
                event['terms'] = value[5].replace('\n', '')
                event_list.append(event)
            sorted_list = sorted(event_list, key=lambda k: k['timestamp'])
            for event in sorted_list:
                event_sorted.write(str(event['review']) + '\t' + str(event['star']) + '\t' + str(event['timestamp']) + '\t' + str(event['user']) + '\t' + str(event['business']) + '\t' + str(event['terms']) + '\n')

# def split_event(train_p, valid_p, test_p):
#     with open(event_sorted_root, 'r') as event_sorted:
#         num_lines = sum(1 for line in event_sorted)
#     with open(event_sorted_root, 'r') as event_sorted:
#         with open(event_sorted_trainroot, 'w') as event_train:
#             with open(event_sorted_validroot, 'w') as event_valid:
#                 with open(event_sorted_testroot, 'w') as event_test:
#                     cur_num = 0
#                     for line in event_sorted:
#                         if cur_num < int(train_p*num_lines):
#                             event_train.write(line)
#                         elif int(train_p*num_lines) <= cur_num < int((train_p+valid_p)*num_lines):
#                             event_valid.write(line)
#                         else:
#                             event_test.write(line)
#                         cur_num += 1

def contain_zh(word):
    zh_pattern = re.compile(u'[\u4e00-\u9fff]+')
    match = zh_pattern.search(word)
    return match

def search_ch():
    with open(event_data_root, 'r') as event_data:
        for line in event_data:
            value = line.split('\t')
            text = value[4]
            if contain_zh(text):
                print(text)

if __name__ == '__main__':
    # search_ch()
    change_eventwithid()
    sort_event()
    # split_event(0.8, 0.1, 0.1)



