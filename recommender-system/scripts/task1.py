from pyspark import SparkContext
from collections import defaultdict
import random
from itertools import combinations
import csv
import sys


def hashing(user):  # Define Hash Function
    my_list = []
    m = len(users)
    for i in range(100):  # 100 is the number of hash functions
        a = random.randint(1, 935678)
        b = random.randint(1, 986464)
        my_list.append((a * user + b) % m)
    return my_list


def get_index(x):  # Indexing the entry in order to hash
    return indexes[x]


def assign_same_bucket(signa, buckets, row):
    lsh = list()
    for i in range(buckets):
        same = defaultdict(set)
        for business_id, hashing_res in signa:
            pos = i * row
            value = tuple(hashing_res[pos:pos + row])
            same[value].add(business_id)
        for k, v in same.items():
            if len(v) <= 1:
                continue
            else:
                list_of_pairs = combinations(v, 2)
                for j in list_of_pairs:
                    lsh.append(tuple(sorted(j)))
    lsh = sorted(list(set(lsh)))
    return lsh


def jaccard_sim(candidate, business_user_map):
    user_set_0 = business_user_map[candidate[0]]
    user_set_1 = business_user_map[candidate[1]]
    jac = len(user_set_0.intersection(user_set_1))/len(user_set_0.union(user_set_1))
    return jac


if __name__ == "__main__":
    sc = SparkContext('local[*]', 'task1')

    input_path = sys.argv[1]
    out_file = sys.argv[2]
    header = sc.textFile(input_path).map(lambda x: x.split(',')).first()
    train_rdd = sc.textFile(input_path).map(lambda x: x.split(',')).filter(lambda x: x != header).\
        map(lambda x: (x[0], x[1]))
    # print(train_rdd.collect())   # This train rdd only contains (user_id, business_id), since all data are scored
    users = train_rdd.map(lambda x: x[0]).distinct().collect()  # All the users, to make signature matrix
    businesses = train_rdd.map(lambda x: x[1]).distinct().collect()  # All the businesses
    # print(users)
    indexes = defaultdict(int)
    for index, value in enumerate(users):
        indexes[value] = index
    # print(indexes)
    user_business = train_rdd.map(lambda x: (get_index(x[0]), x[1])).map(lambda x: (x[1], x[0]))
    # print(user_business.collect())

    # Start hashing. To generate signature we need a list to store result of hashing each time
    min_hashing_storage = defaultdict(list)
    sign_matrix = user_business.groupByKey().mapValues(set)
    # print(sign_matrix.collect())

    for user in indexes.values():  # Get our hashing results for each of the users
        min_hashing_storage[user] = hashing(user)

    # Get the Sign Matrix of the businesses. Included the set of users each business have.
    sign = user_business.map(lambda x: (x[1], x[0])). \
        groupByKey().distinct().map(lambda x: (x[1], x[0])). \
        mapValues(lambda x: min_hashing_storage[x]). \
        map(lambda x: (list(x[0]), x[1])). \
        flatMap(lambda x: ((i, x[1]) for i in x[0])) \
        .groupByKey().mapValues(lambda x: list(x)) \
        .mapValues(lambda x: zip(*x)). \
        mapValues(lambda x: [min(val) for val in x]).collect()

    bucket = 50
    rows = 2
    lsh_candidates = assign_same_bucket(sign, bucket, rows)

    # To compute Jaccard Similarity, we need sets of users that corresponding to business id
    business_user = train_rdd.map(lambda x: (x[1], x[0])).groupByKey().mapValues(set)
    business_user = business_user.collectAsMap()
    # print(business_user)

    with open(out_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['business_id_1', ' business_id_2', ' similarity'])
        for candidate in lsh_candidates:
            jac_sim = jaccard_sim(candidate, business_user)
            if jac_sim >= 0.5:
                writer.writerow([candidate[0], candidate[1], jac_sim])




