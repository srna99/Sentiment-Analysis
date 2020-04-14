"""
Serena Cheng
CMSC 416 - PA 5: Sentiment Analysis (scorer.py)
4/14/2020
~~~~~
Problem:
This program addresses the goal of scoring the accuracy of the decision list model. The overall accuracy of
the implemented model is 0.71 or 71%.
Usage:
The user should include the template below when executing:
    (answers to test data) (test key data)
Ex. python3 scorer.py my-sentiment-answers.txt sentiment-test-key.txt
    > Baseline Accuracy (positive) : 0.64
    > Overall Accuracy: 0.71
    >          positive negative
    >  positive     148       12
    >  negative      56       16
Algorithm:
Each sentiment in the answer set is compared to the sentiments in the key and then inputted in the confusion
matrix. The baseline accuracy is calculated by assuming every sentiment to be the most frequent one (in
this case it is "positive") and dividing the number of correct instances by the total number of answers. The
overall accuracy is calculated by dividing the total correct sentiments by the
total number of answers.
"""

import sys
import re

test_answers = sys.argv[1]  # answer set file
test_key = sys.argv[2]  # test key file

answer_content = []  # test tokenized
key_content = []  # key tokenized

answers = {}  # sentiments in answer
key = {}  # sentiments in key

sentiments = {"positive": 0, "negative": 1}  # associate sentiments with number


# split sentiments from instances and add to respective sentiments dicts
def get_sentiments(answer_list, instance_sentiment_dict):
    for ans in answer_list:
        ans_parts = re.search(r'instance=\"(.*)\" sentiment=\"(.*)\"', ans)
        instance = ans_parts.group(1)
        sentiment = ans_parts.group(2)
        instance_sentiment_dict[instance] = sentiment


# tokenize answers
with open(test_answers, 'r', encoding="utf-8-sig") as file:
    answer_content.extend(file.read().split("\n"))
    answer_content.pop()

# tokenize key
with open(test_key, 'r', encoding="utf-8-sig") as file:
    key_content.extend(file.read().split("\n"))
    key_content.pop()

# get sentiments by instance ids
get_sentiments(answer_content, answers)
get_sentiments(key_content, key)

# create matrix filled with 0s that is sized (num of sentiments x num of sentiments)
confusion_matrix = [[0] * len(sentiments) for _ in range(len(sentiments))]

most_sentiment = ''  # most frequent sentiment
most_count = 0  # correct count of frequent sentiment
total_correct = 0  # total correct count of all sentiments
for instance_id in key.keys():
    actual_sentiment = key[instance_id]  # actual
    predicted_sentiment = answers[instance_id]  # predicted

    # increment on matrix
    confusion_matrix[sentiments[actual_sentiment]][sentiments[predicted_sentiment]] += 1

    if actual_sentiment == predicted_sentiment:
        # update if sentiments count is higher than current highest count
        if confusion_matrix[sentiments[actual_sentiment]][sentiments[predicted_sentiment]] > most_count:
            most_sentiment = actual_sentiment
            most_count = confusion_matrix[sentiments[actual_sentiment]][sentiments[predicted_sentiment]]

        # increment total correct count if sentiment is correct
        total_correct += 1

# BA = (num of correct answers if all are most frequent sentiment)/total num of answers
baseline_accuracy = round(most_count / len(key), 2)
# A = num of correct answers/total num of answers
accuracy = round(total_correct / len(key), 2)

print("Baseline Accuracy (" + most_sentiment + ") :", baseline_accuracy)
print("Overall Accuracy:", accuracy)

ordered_tags = [''] * len(sentiments)
# put sentiments in order
for tag, num in sentiments.items():
    ordered_tags[num] = tag

# print all sentiments horizontally
for ind in range(len(ordered_tags)):
    if ind == 0:
        print("\n         ", end='')
    print(format(ordered_tags[ind], '>4'), end=' ')

# print all sentiments and counts in matrix
for ind in range(len(sentiments)):
    print("\n", format(ordered_tags[ind], '8'), end='')
    for j in range(len(sentiments)):
        print(format(confusion_matrix[ind][j], '8'), end=' ')
