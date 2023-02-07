import json
import operator

business_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business_catefiltered.json'
business_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed_nv/business.dict'
state_cnt = dict()
city_cnt = dict()


def choose_business():
    count = 0
    with open(business_data_root, 'r') as business_data:
        with open(business_out_root, 'w') as fout:
            for line in business_data:
                business = json.loads(line)
                zipCode = business['postal_code']
                # if len(zipCode) != 5:
                #     continue
                if business['review_count'] < 10:
                    continue
                if business['city'] != 'Las Vegas':
                    continue
                count += 1
                try:
                    fout.write(business['business_id'] + '\t'
                           + '|'.join(business['categories']) + '\t'
                           + str(business['review_count']) + '\t'
                           + business['name'] + '\t'
                           + business['city'] + '\t'
                           + business['state']+ '\t'
                           + zipCode+ '\n')
                except:
                    print(business['business_id'])
                    print('|'.join(business['categories']))
                    print(str(business['latitude']))
                    print(str(business['longitude']))
                    print(business['name'])
                    print(business['address'])
                    print('zip', zipCode)
                    print(business['city'])
                    print(business['state'])

                state = business['state']
                state_cnt[state] = state_cnt.get(state, 0) + 1
                city = business['city']
                city_cnt[city] = city_cnt.get(city, 0) + 1

    print(count)
    sorted_state = sorted(state_cnt.items(), key=operator.itemgetter(1))
    sorted_state.reverse()
    for st in range(len(state_cnt)):
        print(st, sorted_state[st])

    sorted_cate = sorted(city_cnt.items(), key=operator.itemgetter(1))
    sorted_cate.reverse()
    for i in range(30):
        print(sorted_cate[i][0], sorted_cate[i][1])

if __name__ == '__main__':
    choose_business()

