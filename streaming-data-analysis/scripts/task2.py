from blackbox import BlackBox
import random
import binascii
from collections import defaultdict
import sys
from statistics import median
import csv


def myhashs(user):  # Define Hash Function
    my_list = []
    m = 100000000
    user = int(binascii.hexlify(user.encode('utf8')), 16)
    for i in range(2000):  # 500 is the number of hash functions
        a = random.randint(1, 923564789)
        b = random.randint(1, 938475749)
        my_list.append((a * user + b) % m)
    return my_list


def trailing_zero(user):
    without_zero = len(str(user).rstrip('0'))
    trail_zero = len(str(user)) - without_zero
    return trail_zero


if __name__ == "__main__":
    bx = BlackBox()
    input_file = sys.argv[1]
    num_ask = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    output_file = sys.argv[4]
    num_group = 4
    group_length = 500  # #hash functions/#groups
    record = defaultdict(int)

    for iteration in range(num_ask):
        max_trail_per_group = [[0 for i in range(group_length)] for j in range(num_group)]
        for user_id in bx.ask(input_file, stream_size):
            hashed_user = myhashs(user_id)
            for i, hashed_value in enumerate(hashed_user):
                group_number = i // group_length
                pos_in_group = i % group_length
                zeros_num = trailing_zero(format(hashed_value, 'b'))  # binary value of hashing
                max_trail_per_group[group_number][pos_in_group] = max(zeros_num,
                                                                      max_trail_per_group[group_number][pos_in_group])
        group_avg = []
        for group in max_trail_per_group:
            average = sum(group)/len(group)
            group_avg.append(2 ** average)
        med = median(group_avg)
        record[iteration] = int(med)

    with open(output_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Time', 'Ground Truth', 'Estimation'])
        for k, v in record.items():
            writer.writerow([str(k), '300', str(v)])


