import json
import business_restaurants
import user
import term
import review
# import

review_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/review.json'
user_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/user.json'
business_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/dataset/business.json'

if __name__=="__main__":
    business_restaurants.choose_business()
    # choose_user(user_root)
    # generate_term(review_root)
    # generate_event(review_root)
