import json
import random

data = open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/business.json')
state_cnt = dict()
cuisine_cnt = dict()
cuisine_set = dict()

target = ['mexican', 'chinese', 'italian', 'american (traditional)', 'american (new)', 'thai', 'japanese', 'vietnamese', 'indian']

for t in target:
    cuisine_set[t] = []

target = set(target)
for line in data:
    jdata = json.loads(line)
    cate = jdata['categories']

    if 'Restaurants' not in cate:
        continue
    if len(cate) > 2:
        continue
    if jdata['review_count'] < 50:
        continue

    zipCode = jdata['postal_code']
    if len(zipCode) != 5:
        continue

    Flag = True
    for c in cate:
        x = c.lower()
        if x == 'restaurants':
            continue
        if x not in target:
            Flag = False
            break
        cuisine_cnt[x] = cuisine_cnt.get(x, 0) + 1
        cuisine_set[x].append(jdata['business_id'])

    if Flag == False:
        continue

fout = open('/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/business.label', 'w')
for x in cuisine_set:
    try:
        res = random.sample(cuisine_set[x], 100)
    except:
        print(x)
        print(cuisine_cnt)
    for i in res:
        fout.write('%s\t%s\n' % (i, x))
fout.close()

for x in cuisine_cnt:
    print(x, cuisine_cnt[x])




