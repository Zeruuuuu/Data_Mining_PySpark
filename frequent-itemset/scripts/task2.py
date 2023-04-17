from collections import defaultdict
import sys
import time
from itertools import combinations
import math
from pyspark import SparkContext, SparkConf
import csv


def get_single_elements(data_flow):   # To begin with, we need to initiallize our Apriori. We need single elements
    single_element = defaultdict(int)
    for List in data_flow:
        for Sets in List:
            single_element[Sets] += 1
    #print(single_element)
    return single_element


# Function that given a dictionary of candidates and their count, we filter out using local support
# This is for single elements only. For the following steps it is easier to combine together
def get_local_candidates(dic, sup):
    candidate_list = sorted([value for value, count in dic.items() if count >= sup])
    return candidate_list


# With our previous frequent candidates, we need to rule out all element that have subset not belong to previous
def update_candidate_list(data_flow, prev, k, sup):
    multiple_element = defaultdict(int)
    for List in data_flow:
        potential_ans = set(List).intersection(prev)
        selected_ls = sorted(potential_ans)
        for i in combinations(selected_ls, k):
            multiple_element[tuple(i)] += 1
    candidate_list = sorted([value for value, count in multiple_element.items() if count >= sup])
    return candidate_list


def apriori(data_flow, local_sup):
    frequent = []
    data_list = list(data_flow)
    singles = get_single_elements(data_list)
    singles_candidates = get_local_candidates(singles, local_sup)
    single_temp_list = []
    for single in singles_candidates:
        single_temp_list.append((single,))    # Since we want all to be equal format, for singles, we also make them
    frequent.append(single_temp_list)         # Tuples format
    k = 2
    while_loop_standard = singles_candidates
    while while_loop_standard:
        updated_candidate = update_candidate_list(data_list, while_loop_standard, k, local_sup)
        frequent.append(updated_candidate)
        while_loop_standard = set().union(*updated_candidate)
        k += 1
    return frequent


def SON_phase_2(data_flow, candidates):
    res = defaultdict(int)
    for basket in data_flow:
        for List in candidates:
            if set(List).issubset(basket):
                res[List] += 1
    freq_pairs = []
    for value, count in res.items():
        freq_pairs.append((value, count))
    return freq_pairs


if __name__ == '__main__':
    start = time.time()
    conf = SparkConf().setAppName('task2')
    sc = SparkContext(conf=conf)

    k = int(sys.argv[1])
    support = int(sys.argv[2])
    input_path = sys.argv[3]
    output_path = sys.argv[4]
    rdd = sc.textFile(input_path)
    col_1 = rdd.first()
    processed_rdd = rdd.filter(lambda x: x != col_1).\
        map(lambda x: x.replace('"', '').split(',')).\
        map(lambda x: (x[0], x[1], x[5])).\
        map(lambda x: (x[0][:-4] + x[0][-2:], x[1], x[2])).\
        map(lambda x: (x[0] + '-' + x[1].lstrip('0'), x[2].lstrip('0'))).distinct()

    header = ['DATE-CUSTOMER_ID', 'PRODUCT_ID']
    data = processed_rdd.collect()
    data.insert(0, header)
    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

    # Read Data
    my_rdd = sc.textFile('/Users/zhouzeru/Desktop/553hw1/2/scripts/out.csv').map(lambda x: x.split(','))
    header = my_rdd.first()
    my_rdd = my_rdd.filter(lambda x: x != header).\
        map(lambda x: (x[0], x[1])).\
        groupByKey().\
        filter(lambda x: len(x[1]) > k).map(lambda x: list(set(x[1])))

    partitions = my_rdd.getNumPartitions()  # Get number of Partitions
    single_element = my_rdd.flatMap(lambda x: x). \
        distinct().map(lambda x: {x}).collect()  # Notice that set() and {} are different. I made a lot mistakes here
    local_sup = math.ceil(support / partitions)

    # Phase 1
    candidate = my_rdd.mapPartitions(lambda x: apriori(x, local_sup)) \
        .flatMap(lambda x: x) \
        .distinct() \
        .collect()

    # print(candidate)
    # Phase 2
    frequent_items = my_rdd.mapPartitions(lambda x: SON_phase_2(x, candidate)) \
        .reduceByKey(lambda a, b: a + b) \
        .filter(lambda x: x[1] >= support) \
        .map(lambda x: x[0]) \
        .collect()

    frequent = sorted(frequent_items)
    candidates = sorted(candidate)
    frequent = [i[0] if len(i) == 1 else i for i in frequent]
    candidates = [i[0] if len(i) == 1 else i for i in candidates]

    # Write the file
    single_freq = []
    tuples_freq = []
    single_cand = []
    tuples_cand = []
    for item in frequent:
        if isinstance(item, str):
            single_freq.append(item)
        else:
            tuples_freq.append(item)
    single_freq = sorted(single_freq)
    tuples_freq = sorted(tuples_freq)
    for item in candidates:
        if isinstance(item, str):
            single_cand.append(item)
        else:
            tuples_cand.append(item)
    single_cand = sorted(single_cand)
    tuples_cand = sorted(tuples_cand)

    with open(output_path, 'w') as outfile:
        # For Candidate Set
        outfile.write('Candidates:' + '\n')
        for i, string in enumerate(single_cand):
            if i + 1 == len(single_cand):
                outfile.write("('" + string + "')" + '\n')
            else:
                outfile.write("('" + string + "')" + ',')
        outfile.write('\n')

        max_len = max([len(i) for i in tuples_cand])
        k = 2
        while k <= max_len:
            ele_list = []
            for t in tuples_cand:
                if len(t) == k:
                    ele_list.append(str(t))
            line = ','.join(ele_list)
            outfile.write(line)
            k += 1
            outfile.write('\n')
            outfile.write('\n')

        # For Frequent set
        outfile.write('Frequent Itemsets:' + '\n')
        for i, string in enumerate(single_freq):
            if i + 1 == len(single_freq):
                outfile.write("('" + string + "')" + '\n')
            else:
                outfile.write("('" + string + "')" + ',')
        outfile.write('\n')

        max_len_f = max([len(i) for i in tuples_freq])
        k = 2
        while k <= max_len_f:
            fre_list = []
            for t in tuples_freq:
                if len(t) == k:
                    fre_list.append(str(t))
            line_f = ','.join(fre_list)
            outfile.write(line_f)
            k += 1
            outfile.write('\n')
            outfile.write('\n')
    end = time.time()
    duration = end - start
    print(f'Duration:{end - start}')