import re

review_selected_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review.selected'
auto_phrases_input_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/review.corpus_autophrase'
auto_phrases_output_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/AutoPhrase.txt'
auto_phrases_segmented_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/segmentation.txt'
auto_phrases_segmented_out = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/out.txt'
event_out_root = '/home1/wyf/Projects/dynamic_network_embedding/data/yelp/processed/event.txt'

def generate_corpus():
    with open(review_selected_root, 'r') as review_out:
        with open(auto_phrases_input_root, 'w') as input:
            for line in review_out:
                value = line.split('\t')
                review = value[4]
                input.write(review + '\n')

def generate_term():
    terms_weight_dict = dict()
    review_terms_dict = dict()
    with open(auto_phrases_output_root, 'r') as auto_phrases:
        for line in auto_phrases:
            value = line.split('\t')
            terms_weight_dict[value[1].replace('\n', '')] = float(value[0])

    count = 0
    with open(auto_phrases_segmented_root, 'r') as segmented_data:
        with open(auto_phrases_segmented_out, 'w') as out:
            pattern = r'<phrase>(.*?)</phrase>'
            for line in segmented_data:
                count += 1
                terms_tmp = re.findall(pattern, line, re.S|re.M)
                terms = []
                for term in terms_tmp:
                    if term not in terms_weight_dict:
                        print(term)
                        continue
                    if terms_weight_dict[term] > 0.8:
                        if term not in terms:
                            terms.append(term)
                # terms = list(set(terms))
                text = ';'.join(terms)
                review_terms_dict[count] = terms
                out.write(text + '\n')

    return review_terms_dict

def check_term():
    terms_weight_dict = dict()
    term_frequency_dict1 = dict()
    term_frequency_dict2 = dict()

    with open(auto_phrases_output_root, 'r') as auto_phrases:
        for line in auto_phrases:
            value = line.split('\t')
            terms_weight_dict[value[1].replace('\n', '')] = float(value[0])

    count = 0
    count1 = 0
    count2 = 0
    with open(auto_phrases_segmented_root, 'r') as segmented_data:
        pattern = r'<phrase>(.*?)</phrase>'
        for line in segmented_data:
            count += 1
            terms = []
            terms_tmp = re.findall(pattern, line, re.S | re.M)
            if terms_tmp:
                count1 += 1
            for term in terms_tmp:
                if term not in term_frequency_dict1:
                    term_frequency_dict1[term] = 1
                else:
                    term_frequency_dict1[term] += 1

                if term not in terms_weight_dict:
                    continue
                if terms_weight_dict[term] > 0.8:
                    terms.append(term)
                    if term not in term_frequency_dict2:
                        term_frequency_dict2[term] = 1
                    else:
                        term_frequency_dict2[term] += 1
            if terms:
                count2 += 1

    print(len(term_frequency_dict1))
    # print([k for k,i in term_frequency_dict1.items() if i<10])
    print(len([i for i in term_frequency_dict2.values() if i>=10]))
    print(len(term_frequency_dict2))
    print(min(term_frequency_dict2.values()))
    print(count)
    print(count1)
    print(count2)
    # print(term_frequency_dict1)
    # print(term_frequency_dict2)
    selected_term = [k for k,i in term_frequency_dict2.items() if i>=10]
    return selected_term


def generate_event(review_terms_dict, selected_term):
    count = 0
    with open(review_selected_root, 'r') as review_out:
        with open(event_out_root, 'w') as event_out:
            for line in review_out:
                count += 1
                review_terms = review_terms_dict[count]
                terms = [i for i in review_terms if i in selected_term]
                if not terms:
                    continue
                term_text = ';'.join(terms)
                line = line.replace('\n', '\t') + term_text + '\n'
                event_out.write(line)


if __name__ == '__main__':
    # generate_corpus()
    review_terms_dict = generate_term()
    selected_term = check_term()
    generate_event(review_terms_dict, selected_term)
