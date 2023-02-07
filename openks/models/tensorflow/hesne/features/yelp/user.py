import json

user_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/user.json'
user_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/user.dict'


def choose_user():
    count = 0
    user_dict = dict()
    with open(user_data_root, 'r') as user_data:
        with open(user_out_root, 'w') as fout:
            for line in user_data:
                user = json.loads(line)
                if user['review_count'] < 10:
                    continue
                count += 1

                try:
                    fout.write(user['user_id'] + '\t' + user['name'] + '\t' + str(user['review_count']) + '\n')
                    user_dict[user['user_id']] = user['name']
                except:
                    print('user_id:', user['user_id'])
                    print('name:', user['name'])
                    print('review_count:', str(user['review_count']))
    print(count)

if __name__ == '__main__':
    choose_user()




