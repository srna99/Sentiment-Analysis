"""
Serena Cheng
CMSC 416 - PA 5: Sentiment Analysis (sentiments.py)
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
w-1, w-2 & w-1, w+1 & w+2, and several more. Smoothing was applied, where each feature's sentiments frequency
was increased by 1. Tests based on the features were created and ranked based on their log-likelihood score
(abs(log( (P(sense_1|feature)/(P(sense_2|feature) ))). The tests are then applied to the test data in
ranked order, where the first test to pass results in the associated sentiments.
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

# sentiments
sentiment_positive = "positive"
sentiment_negative = "negative"

feature_sense_dict = {}  # sentiments frequency based on features
feature_frequency_dict = {}  # feature frequency
sense_frequency_dict = {sentiment_positive: 0, sentiment_negative: 0}  # sentiments frequency

ranked_tests = []  # ranked tests
answers = []  # answers to test set


# gets rid of unnecessary elements in context and returns a tokenized version
def clean_context(content):
    content = re.sub(r'(<.>|<\/.>|\.|,|;|!|-|&|\)|\(|\"|\?)', ' ', content)
    content = re.sub(r'\s+', ' ', content)
    return content.split()


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
    # get sentiments and update count
    sentiment = re.search(r'sentiment=\"(.*)\"', instance).group(1)
    sense_frequency_dict[sentiment] += 1

    # get context then clean and tokenize it
    context = re.search(r'<context>\n(.*)\n</context>', instance).group(1)
    context_words = clean_context(context)

    # get features
    for feature_word in context_words:
        if feature_word not in feature_sense_dict.keys():
            feature_sense_dict[feature_word] = {sentiment_positive: 1, sentiment_negative: 1}

        feature_sense_dict[feature_word][sentiment] += 1

        if feature_word in feature_frequency_dict.keys():
            feature_frequency_dict[feature_word] += 1
        else:
            feature_frequency_dict[feature_word] = 1

# calculate log-likelihood ratio for each feature
for feature, sentiment_frequencies in feature_sense_dict.items():
    # probability of phone given feature
    prob_pos = sentiment_frequencies[sentiment_positive] / feature_frequency_dict[feature]
    # probability of product given feature
    prob_neg = sentiment_frequencies[sentiment_negative] / feature_frequency_dict[feature]

    # pick sentiments with higher probability
    if prob_pos >= prob_neg:
        sentiment = sentiment_positive
    else:
        sentiment = sentiment_negative

    # abs(log( (P(sense_1|feature)/(P(sense_2|feature) ))
    log_likelihood_ratio = round(abs(math.log(prob_pos / prob_neg)), 5)

    # add calculated ratio, feature type, feature, and sentiments to test
    test = (log_likelihood_ratio, feature, sentiment)
    ranked_tests.append(test)

# sort tests in descending order
ranked_tests.sort(key=lambda x: x[0], reverse=True)

# write info to model log
with open(model, 'w', encoding="utf-8-sig") as file:
    for data in ranked_tests:
        line = "feature = " + data[1] + ", sentiment = " + data[2] + ", log-likelihood score = " + str(
                data[0]) + "\n"
        file.write(line)

# for test set
for instance in test_content:
    # get instance id
    instance_id = re.search(r'instance id=\"(.*)\"', instance).group(1)

    # get context then clean and tokenize it
    context = re.search(r'<context>\n(.*)\n</context>', instance).group(1)
    context_words = clean_context(context)

    sentiment = ""  # chosen sentiments
    # going through tests in order and if there is a success, assign associated sentiments to target
    for test in ranked_tests:
        # if training feature is in test then success
        if test[1] in context_words:
            sentiment = test[2]

    # if no sentiments was assigned and every test failed, assign the most frequent sentiments to target
    if sentiment == "":
        if sense_frequency_dict[sentiment_positive] >= sense_frequency_dict[sentiment_negative]:
            sentiment = sentiment_positive
        else:
            sentiment = sentiment_negative

    # format answer and add to list
    answer = "<answer instance=\"" + instance_id + "\" sentiment=\"" + sentiment + "\"/>"
    answers.append(answer)

# print answers
for answer in answers:
    print(answer)
