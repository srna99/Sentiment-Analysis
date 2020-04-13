"""
Serena Cheng
CMSC 416 - PA 5: Sentiment Analysis (sentiment.py)
4/14/2020
~~~~~
Problem:
This program aims to disambiguate the words "line/lines" through the decision list model. Selected features
were chosen to gain contextual information about the targeted word.
Usage:
The user should include additional arguments when executing program as follows:
    (annotated training data) (test data) (model log) > (answers)
Ex. python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
    > <answer instance="line-n.w8_053:3883:" senseid="phone"/> etc.
Algorithm:
This model is based on Yarowsky's decision list. Unigrams and bi-grams surrounding the target word are
collected as features and their position relative to the target word is noted. Features used includes w+1,
w-1, w-2 & w-1, w+1 & w+2, and several more. Smoothing was applied, where each feature's sense frequency
was increased by 1. Tests based on the features were created and ranked based on their log-likelihood score
(abs(log( (P(sense_1|feature)/(P(sense_2|feature) ))). The tests are then applied to the test data in
ranked order, where the first test to pass results in the associated sense.
Baseline Accuracy (phone): 0.51
Overall Accuracy: 0.9
        phone product
 phone     64     8
 product    4    50
"""

import sys
import re
import math

train_set = sys.argv[1]  # training set file
test_set = sys.argv[2]  # test set file
model = sys.argv[3]  # model log file

train_content = []  # tokenized training set
test_content = []  # tokenized test set

# senses
sense_phone = "phone"
sense_product = "product"

# feature types
w_plus_1 = "+1"  # w+1
w_minus_1 = "-1"  # w-1
w_pair_minus_2 = "-2 & -1"  # w-2 & w-1
w_pair_plus_2 = "+1 & +2"  # w+1 & w+2
w_pair_minus_plus_1 = "-1 & +1"  # w-1 & w+1
w_window_3 = "k=3"  # +=3
w_window_5 = "k=5"  # +=5
w_window_10 = "k=10"  # +=10

# sense frequency based on features
feature_sense_dict = {w_plus_1: {}, w_minus_1: {}, w_pair_minus_2: {}, w_pair_plus_2: {},
                      w_pair_minus_plus_1: {}, w_window_3: {}, w_window_5: {}, w_window_10: {}}
feature_frequency_dict = {}  # feature frequency
sense_frequency_dict = {sense_phone: 0, sense_product: 0}  # sense frequency

ranked_tests = []  # ranked tests
answers = []  # answers to test set


# gets rid of unnecessary elements in context and returns a tokenized version
def clean_context(content):
    content = re.sub(r'(<.>|<\/.>|\.|,|;|!|-|&|\)|\(|\"|\?)', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    content = re.sub(">lines<", ">line<", content)
    return content.split()


# adds and updates counts of senses according to features
def add_to_feature_sense_dict(w_feature, feature_word):
    if feature_word not in feature_sense_dict[w_feature].keys():
        feature_sense_dict[w_feature][feature_word] = {sense_phone: 1, sense_product: 1}

    feature_sense_dict[w_feature][feature_word][sense] += 1


# gets all unigrams within a window and updates frequencies
def add_with_window_size(context_list, w_feature, k, target_ind, add_frequencies):
    for ind in range(target_ind - k, target_ind + k):
        # skip index if out of range or is target
        if ind < 0 or ind >= len(context_list) or ind == target_ind:
            continue

        add_to_feature_sense_dict(w_feature, context_list[ind])

        # updates frequencies
        if add_frequencies:
            increment_feature_frequency(context_list[ind])


# adds and updates counts of features
def increment_feature_frequency(feature_term):
    if feature_term in feature_frequency_dict.keys():
        feature_frequency_dict[feature_term] += 1
    else:
        feature_frequency_dict[feature_term] = 1


# creates test based on info from ranked tests list and returns if the test was a success or a failure
def create_test(test_number, feature_term, is_list):
    test_feature = ranked_tests[test_number]  # test info according to number

    # if training feature is in tested feature list then success
    if is_list and test_feature[2] in feature_term:
        return True
    # if tested feature is same as training feature then success
    elif not is_list and feature_term == test_feature[2]:
        return True

    # if test failed then failure
    return False


# creates list of features of window size from test data
def create_test_feature_list(context_list, k, target_ind):
    features_list = []
    for i in range(target_ind - k, target_ind + k):
        # skip if out of range or is target
        if i < 0 or i >= len(context_list) or i == target_ind:
            continue
        features_list.append(context_list[i])

    return features_list


# tokenize training set by instance and pop off useless last bit
with open(train_set, 'r', encoding="utf-8-sig") as file:
    train_content.extend(file.read().lower().split("</instance>"))
    train_content.pop()

# tokenize test set by instance and pop off useless last bit
with open(test_set, 'r', encoding="utf-8-sig") as file:
    test_content.extend(file.read().lower().split("</instance>"))
    test_content.pop()

# for training set
for instance in train_content:
    # get sense and update count
    sense = re.search(r'senseid=\"(.*)\"', instance).group(1)
    sense_frequency_dict[sense] += 1

    # get context then clean and tokenize it
    context = re.search(r'<context>\n(.*)\n</context>', instance).group(1)
    context_words = clean_context(context)

    target = context_words.index("<head>line</head>")  # index of target

    # get features of type w+1
    if target + 1 < len(context_words):
        add_to_feature_sense_dict(w_plus_1, context_words[target + 1])

    # get features of type w-1
    if target - 1 >= 0:
        add_to_feature_sense_dict(w_minus_1, context_words[target - 1])

    # get features of type w-2 & w-1
    if target - 2 >= 0:
        bigram = context_words[target - 2] + " " + context_words[target - 1]
        add_to_feature_sense_dict(w_pair_minus_2, bigram)
        increment_feature_frequency(bigram)

    # get features of type w+1 & w+2
    if target + 2 < len(context_words):
        bigram = context_words[target + 1] + " " + context_words[target + 2]
        add_to_feature_sense_dict(w_pair_plus_2, bigram)
        increment_feature_frequency(bigram)

    # get features of type w-1 & w+1
    if target - 1 >= 0 and target + 1 < len(context_words):
        bigram = context_words[target - 1] + " " + context_words[target + 1]
        add_to_feature_sense_dict(w_pair_minus_plus_1, bigram)
        increment_feature_frequency(bigram)

    # get features of window sizes 3, 5, and 10
    add_with_window_size(context_words, w_window_3, 3, target, False)
    add_with_window_size(context_words, w_window_5, 5, target, False)
    add_with_window_size(context_words, w_window_10, 10, target, True)

# calculate log-likelihood ratio for each feature
for feature_type, feature_list in feature_sense_dict.items():
    for feature, frequency in feature_list.items():
        # probability of phone given feature
        prob_phone = frequency[sense_phone] / feature_frequency_dict[feature]
        # probability of product given feature
        prob_product = frequency[sense_product] / feature_frequency_dict[feature]

        # pick sense with higher probability
        if prob_phone >= prob_product:
            sense = sense_phone
        else:
            sense = sense_product

        # abs(log( (P(sense_1|feature)/(P(sense_2|feature) ))
        log_likelihood_ratio = round(abs(math.log(prob_phone / prob_product)), 5)

        # add calculated ratio, feature type, feature, and sense to test
        test = (log_likelihood_ratio, feature_type, feature, sense)
        ranked_tests.append(test)

# sort tests in descending order
ranked_tests.sort(key=lambda x: x[0], reverse=True)

# write info to model log
with open(model, 'w', encoding="utf-8-sig") as file:
    for data in ranked_tests:
        line = "feature type = " + data[1] + ", feature = " + data[2] + ", log-likelihood score = " + str(
                data[0]) + ", sense = " + data[3] + "\n"
        file.write(line)

# for test set
for instance in test_content:
    # get instance id
    instance_id = re.search(r'instance id=\"(.*)\"', instance).group(1)

    # get context then clean and tokenize it
    context = re.search(r'<context>\n(.*)\n</context>', instance).group(1)
    context_words = clean_context(context)

    target = context_words.index("<head>line</head>")  # index of target

    sense = ""  # chosen sense
    # going through tests in order and if there is a success, assign associated sense to target and break
    for ind, test in enumerate(ranked_tests):
        if test[1] == w_plus_1 and target + 1 < len(context_words):
            if create_test(ind, context_words[target + 1], False):
                sense = test[3]
                break
        elif test[1] == w_minus_1 and target - 1 >= 0:
            if create_test(ind, context_words[target - 1], False):
                sense = test[3]
                break
        elif test[1] == w_pair_minus_2 and target - 2 >= 0:
            if create_test(ind, context_words[target - 2] + " " + context_words[target - 1], False):
                sense = test[3]
                break
        elif test[1] == w_pair_plus_2 and target + 2 < len(context_words):
            if create_test(ind, context_words[target + 1] + " " + context_words[target + 2], False):
                sense = test[3]
                break
        elif test[1] == w_pair_minus_plus_1 and target - 1 >= 0 and target + 1 < len(context_words):
            if create_test(ind, context_words[target - 1] + " " + context_words[target + 1], False):
                sense = test[3]
                break
        elif test[1] == w_window_3:
            if create_test(ind, create_test_feature_list(context_words, 3, target), True):
                sense = test[3]
                break
        elif test[1] == w_window_5:
            if create_test(ind, create_test_feature_list(context_words, 5, target), True):
                sense = test[3]
                break
        elif test[1] == w_window_10:
            if create_test(ind, create_test_feature_list(context_words, 10, target), True):
                sense = test[3]
                break

    # if no sense was assigned and every test failed, assign the most frequent sense to target
    if sense == "":
        if sense_frequency_dict[sense_phone] >= sense_frequency_dict[sense_product]:
            sense = sense_phone
        else:
            sense = sense_product

    # format answer and add to list
    answer = "<answer instance=\"" + instance_id + "\" senseid=\"" + sense + "\"/>"
    answers.append(answer)

# print answers
for answer in answers:
    print(answer)
