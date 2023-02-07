import json
import operator
import time

category_data_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/business.json'
category_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business_catefiltered.json'
cate_dict = dict()

'''
American(Traditional)
Italian
Mexican
Chinese
American(New)
Japanese
Canadian (New)
Indian
Thai
Greek
French
Vietnamese
Korean
German
Latin American
British
Hawaiian
Spanish
Turkish
not in Asian Fusion
not in Mediterranean
not in Middle Eastern
'''

# target = ['mexican', 'chinese', 'italian', 'american (traditional)', 'american (new)', 'thai', 'french', 'japanese', 'vietnamese', 'indian']
def choose_category():
    with open(category_data_root, 'r') as cate_data:
        with open(category_out_root, 'w') as fout:
            for line in cate_data:
                business = json.loads(line)
                cate = business['categories']

                if 'Restaurants' not in cate:
                    continue

                # if business['review_count'] < 10:
                #     continue

                # if len(cate)>2:
                #     # print(cate)
                #     # time.sleep(10)
                #     continue

                line = json.dumps(business) + '\n'
                fout.write(line)

                for x in cate:
                    cate_dict[x] = cate_dict.get(x, 0) + 1

    sorted_cate = sorted(cate_dict.items(), key=operator.itemgetter(1))
    sorted_cate.reverse()
    return sorted_cate

if __name__ == '__main__':
    sorted_cate = choose_category()
    for i in range(30):
        print(sorted_cate[i][0], sorted_cate[i][1])