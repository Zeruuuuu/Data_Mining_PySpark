from blackbox import BlackBox
import random
import binascii
from collections import defaultdict
import sys
import csv


def myhashs(user):  # Define Hash Function
    my_list = []
    m = 69997
    user = int(binascii.hexlify(user.encode('utf8')), 16)
    for i in range(60):  # 100 is the number of hash functions
        a = random.randint(1, 923564)
        b = random.randint(1, 938475)
        my_list.append((a * user + b) % m)
    return my_list


def check_user(user, bit_map):
    value_position = myhashs(user)
    for i in value_position:
        if bit_map[i] == 0:
            return False
    return True


def update_bit_map(user, bit_map):
    value_position = myhashs(user)
    for i in value_position:
        bit_map[i] = 1


if __name__ == "__main__":
    input_file = sys.argv[1]
    num_ask = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    output_file = sys.argv[4]
    bx = BlackBox()
    # print(bx.ask(input_file, 15))
    previous_user = set()
    filter_bit_array = [0] * 69997
    record = defaultdict(float)
    for iteration in range(num_ask):  # Start iteration
        stream = bx.ask(input_file, stream_size)
        false_positive = 0
        negative = 0
        for user_id in stream:  # If user is checked but not in previous set, it is false positive.
            if check_user(user_id, filter_bit_array) and user_id not in previous_user:
                false_positive += 1
                negative += 1
            elif (not check_user(user_id, filter_bit_array)) and (user_id not in previous_user):
                negative += 1
            update_bit_map(user_id, filter_bit_array)  # update the filter bit array for future user
            previous_user.add(user_id)  # update the previous user set
        fpr = false_positive/negative
        record[iteration] = fpr
    with open(output_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Time', 'FPR'])
        for k, v in record.items():
            writer.writerow([str(k), str(v)])