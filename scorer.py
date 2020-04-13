"""
Serena Cheng
CMSC 416 - PA 5: Sentiment Analysis (scorer.py)
4/14/2020
~~~~~
Problem:
This program addresses the goal of scoring the accuracy of the decision list model.
The overall accuracy of the implemented model is 0.9 or 90%.
Usage:
The user should include the template below when executing:
    (answers to test data) (test key data)
Ex. python3 scorer.py my-line-answers.txt line-key.txt
    > Baseline Accuracy (phone): 0.51
    > Overall Accuracy: 0.9
    >        phone product
    > phone     64     8
    > product    4    50
Algorithm:
Each sense in the answers set is compared to the sense in the key and then
inputted in the confusion matrix. The baseline accuracy is calculated by
assuming every sense to be the most frequent one (in this case it is "phone") and
dividing the number of correct instances by the total number of answers. The
overall accuracy is calculated by dividing the total correct senses by the
total number of answers.
"""

import sys
import re

line_answers = sys.argv[1]  # answer set file
line_key = sys.argv[2]  # test key file

answer_content = []  # test tokenized
key_content = []  # key tokenized

answers = []  # senses in answer
key = []  # senses in key

senses = {"phone": 0, "product": 1}  # associate sense with number


# split senses from instances and add to respective sense lists
def get_senses(answer_list, instance_sense_list):
    for ans in answer_list:
        sense = re.search(r'senseid=\"(.*)\"', ans).group(1)
        instance_sense_list.append(sense)


# tokenize answers
with open(line_answers, 'r', encoding="utf-8-sig") as file:
    answer_content.extend(file.read().split("\n"))
    answer_content.pop()

# tokenize key
with open(line_key, 'r', encoding="utf-8-sig") as file:
    key_content.extend(file.read().split("\n"))
    key_content.pop()

# get only senses
get_senses(answer_content, answers)
get_senses(key_content, key)

# create matrix filled with 0s that is sized (num of senses x num of senses)
confusion_matrix = [[0] * len(senses) for _ in range(len(senses))]

most_sense = ''  # most frequent sense
most_count = 0  # correct count of frequent sense
total_correct = 0  # total correct count of all senses
for ind in range(len(key)):
    actual_sense = key[ind]  # actual
    predicted_sense = answers[ind]  # predicted

    # increment on matrix
    confusion_matrix[senses[actual_sense]][senses[predicted_sense]] += 1

    if actual_sense == predicted_sense:
        # update if sense count is higher than current highest count
        if confusion_matrix[senses[actual_sense]][senses[predicted_sense]] > most_count:
            most_sense = actual_sense
            most_count = confusion_matrix[senses[actual_sense]][senses[predicted_sense]]

        # increment total correct count if sense is correct
        total_correct += 1

# BA = (num of correct answers if all are most frequent sense)/total num of answers
baseline_accuracy = round(most_count / len(key), 2)
# A = num of correct answers/total num of answers
accuracy = round(total_correct / len(key), 2)

print("Baseline Accuracy (phone):", baseline_accuracy)
print("Overall Accuracy:", accuracy)

ordered_tags = [''] * len(senses)
# put senses in order
for tag, num in senses.items():
    ordered_tags[num] = tag

# print all senses horizontally
for ind in range(len(ordered_tags)):
    if ind == 0:
        print("\n        ", end='')
    print(format(ordered_tags[ind], '>4'), end=' ')

# print all senses and counts in matrix
for ind in range(len(senses)):
    print("\n", format(ordered_tags[ind], '7'), end='')
    for j in range(len(senses)):
        print(format(confusion_matrix[ind][j], '5'), end=' ')
