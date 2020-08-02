#!/usr/local/bin/python3
# CSCI B551 Fall 2019
#
# Authors: Aravind Ramalingam aramali Shruti Desai desaish Aniruddha Patil anipatil
#
# based on skeleton code by D. Crandall, 11/2019
#
# ./break_code.py : attack encryption
#

import random
import math
import copy
import sys
import encode
import collections
import string
import time


#Function to decode the encrypted input
def break_code(string, corpus):
    frequency_map, letter_count = get_probability_table(corpus)
    initial_replace_table, initial_rearrange_table = get_start_tables()
    doc = metropolis_hastings(string, frequency_map, letter_count, initial_replace_table, initial_rearrange_table)
    return doc

#Function to generate the next set of replacement and rearrangement tables
def get_next_tables(replace_table, rearrange_table):
    next_replace_table = copy.deepcopy(replace_table)
    next_rearrange_table = copy.deepcopy(rearrange_table)
    if random.uniform(0,1) > 0.02:
        char1 = random.choice(string.ascii_lowercase)
        char2 = random.choice(string.ascii_lowercase)
        next_replace_table[char1], next_replace_table[char2] = replace_table[char2], replace_table[char1]
    else:
        temp = list(range(0,4))
        idx1 = random.choice(temp)
        idx2 = random.choice(temp[:idx1] + temp[idx1+1:])
        next_rearrange_table[idx1], next_rearrange_table[idx2] = rearrange_table[idx2], rearrange_table[idx1]
    return next_replace_table, next_rearrange_table

#Function to calculate the probability of the given document
def get_probability_doc(doc, frequency_map, letter_count):
    doc_prob = 0
    words = doc.split()
    word_prob = [0] * len(words)
    for i, word in enumerate(words):
        for j, letter in enumerate(word):
            if j == 0:
                val = frequency_map[" "][letter] / len(doc)
                word_prob[i] += -math.log(val) if val != 0 else 0
            else:
                val = frequency_map[letter][word[j - 1]] / len(doc)
                word_prob[i] += -math.log(val) if val != 0 else 0
        doc_prob += word_prob[i]
    return doc_prob

#Function that implements the Metropolis-Hastings algorithm
def metropolis_hastings(input, frequency_map, letter_count, replace_table, rearrange_table):
    curr_replace_table = replace_table
    curr_rearrange_table = rearrange_table
    min_prob=get_probability_doc(input, frequency_map, letter_count)
    solution=input
    start = time.time()
    while time.time()-start<=590:

        doc1 = encode.encode(input, curr_replace_table, curr_rearrange_table)
        curr_replace_table_2, curr_rearrange_table_2 = get_next_tables(curr_replace_table, curr_rearrange_table)
        doc2 = encode.encode(input, curr_replace_table_2, curr_rearrange_table_2)

        doc1_prob = get_probability_doc(doc1, frequency_map, letter_count)
        doc2_prob = get_probability_doc(doc2, frequency_map, letter_count)

        if min_prob>doc2_prob:
            min_prob=doc2_prob
            solution=doc2

        if doc2_prob < doc1_prob:
            curr_replace_table = curr_replace_table_2
            curr_rearrange_table = curr_rearrange_table_2
        else:
            if doc2_prob<0:
                ratio = doc2_prob/doc1_prob
            else:
                ratio = doc1_prob/doc2_prob
            
            if random.uniform(0, 1) < (ratio/10):
                curr_replace_table = curr_replace_table_2
                curr_rearrange_table = curr_rearrange_table_2

    return solution

#Function to generate the initial replacement and rearrange tables
def get_start_tables():
    letters = list(range(ord('a'), ord('z') + 1))
    random.shuffle(letters)
    replace_table = dict(zip(map(chr, range(ord('a'), ord('z') + 1)), map(chr, letters)))
    rearrange_table = list(range(0, 4))
    random.shuffle(rearrange_table)
    return replace_table, rearrange_table

#Function to initialize the probability map from training words
def get_probability_table(corpus):
    letters = list(map(chr, range(97, 123))) + [' ']
    frequency_map = {letters[i]: {letters[j]: 0 for j in range(len(letters))} for i in range(len(letters))}
    words = corpus.split()
    for i in range(len(words)):
        for j in range(len(words[i])):
            if j == 0:
                frequency_map[" "][words[i][j]] += 1
            else:
                frequency_map[words[i][j]][words[i][j - 1]] += 1
    return frequency_map, len(corpus)

if __name__ == "__main__":
    start = time.time()
    if (len(sys.argv) != 4):
        raise Exception("usage: ./break_code.py coded-file corpus output-file")

    encoded = encode.read_clean_file(sys.argv[1])
    corpus = encode.read_clean_file(sys.argv[2])
    print("Started decrypting...")
    decoded = break_code(encoded, corpus)
    with open(sys.argv[3], "w") as file:
        print(decoded, file=file)
    print("...and we are done!")
    end = time.time()
    print('This programs took', end-start, 'seconds to complete')
