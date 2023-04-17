import sys
from pyspark import SparkContext
from collections import defaultdict
import math
from itertools import combinations
import csv


# We want to calculate pearson correlation and use it as the weight
def pearson_co(business1, business2):
    user_rating_dict_1 = business_to_user_rating[business1]
    user_rating_dict_2 = business_to_user_rating[business2]
    user_set_1 = set(user_rating_dict_1.keys())
    user_set_2 = set(user_rating_dict_2.keys())
    # avg_all_1 = sum(user_rating_dict_1.values()) / len(user_rating_dict_1.values())
    # avg_all_2 = sum(user_rating_dict_2.values()) / len(user_rating_dict_2.values())
    common_users = user_set_1.intersection(user_set_2)  # Get common users
    rating_1, rating_2 = [], []
    for user in common_users:  # Get the rating of the common users from 2 businesses
        rating_1.append(user_rating_dict_1[user])
        rating_2.append(user_rating_dict_2[user])
    if len(common_users) == 0:
        res = 0
    else:
        avg_1, avg_2 = sum(rating_1) / len(rating_1), sum(rating_2) / len(rating_2)
        reg_rating_1 = [num - avg_1 for num in rating_1]
        reg_rating_2 = [num - avg_2 for num in rating_2]
        above = sum([reg_rating_1[i] * reg_rating_2[i] for i in range(len(reg_rating_1))])
        below = math.sqrt(sum([num ** 2 for num in reg_rating_1])) * math.sqrt(sum([num ** 2 for num in reg_rating_2]))
        if below == 0:
            res = 0
        else:
            pearson_correlation = above / below
            res = pearson_correlation
    return res


def prediction_func(x):  # x in the format of ((user_pred, business_pred), {b1:r1, b2:r2, ...})
    user_pred = x[0][0]
    business_pred = x[0][1]
    avg_rating_business = sum(x[1].values())/len(x[1].values())
    rating = business_to_user_rating[business_pred].values()
    avg_rating_user = sum(rating) / len(rating)
    avg_rating = 0.6 * avg_rating_user + 0.4 * avg_rating_business
    my_dict = x[1]
    if business_pred in my_dict.keys():
        del my_dict[business_pred]
    bus = my_dict.keys()
    bus_update = []
    for b in bus:
        if len(set(business_to_user_rating[b]) & set(business_to_user_rating[business_pred])) >= 10:
            bus_update.append(b)
    bus_final = []
    for b in bus_update:
        pearson = pearson_co(business_pred, b)
        if pearson > 0.5:
            bus_final.append([pearson, b])  # case amplification
    bus_final = sorted(bus_final, reverse=False)[:10]
    above = 0
    below = 0
    for lis in bus_final:
        above += lis[0] * my_dict[lis[1]]
        below += abs(lis[0])
    if below != 0 and above != 0:
        res = above / below
    else:
        res = avg_rating
    return user_pred, business_pred, res


if __name__ == "__main__":
    sc = SparkContext('local[*]', 'task2_1')

    training_input_path = sys.argv[1]
    val_path = sys.argv[2]
    out_path = sys.argv[3]
    # Since it is item-based CF, we want business_id to be the key, and (user, rating) pairs as value
    # We want to filter the users when comparing, use co-rating, so both rating and user are needed
    header = sc.textFile(training_input_path).map(lambda x: x.split(',')).first()
    train_rdd = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: (x[1], (x[0], float(x[2])))).groupByKey().mapValues(dict)
    # print(train_rdd.collect())
    # Now, the format is business_id: dictionary of user-rating pairs. We can load this data to a dictionary
    # and pass it to our pearson correlation function
    business = train_rdd.map(lambda x: x[0]).distinct().collect()  # distinct businesses

    business_to_user_rating = defaultdict(dict)
    data = train_rdd.collect()
    for item in data:
        business_to_user_rating[item[0]] = item[1]
    # print(business_to_user_rating)

    header_val = sc.textFile(val_path).map(lambda x: x.split(',')).first()
    val_rdd = sc.textFile(val_path).map(lambda x: x.split(',')).filter(lambda x: x != header_val)
    # true_res = val_rdd.map(lambda x: float(x[2])).collect()
    val_data = val_rdd.map(lambda x: (x[0], x[1]))  # (user, business)
    # user_business = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header).\
    #     map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict).collectAsMap()
    train_data = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: (x[0], (x[1], float(x[2])))).groupByKey().mapValues(dict)
    full_data = val_data.leftOuterJoin(train_data)  # (user_id, (target_business_id, {b1: r1, b2: r2 ...}))
    # print(len(full_data.collect()))
    dict_cold_start = defaultdict(float)
    # print(full_data.take(5))
    # business-average-rating
    business_ra = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: (x[1], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

    # user-average-rating
    user_ra = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: (x[0], float(x[2]))).groupByKey().mapValues(lambda x: sum(x) / len(x)).collectAsMap()

    # global-average-rating
    rate = sc.textFile(training_input_path).map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: float(x[2])).collect()
    global_avg = sum(rate) / len(rate)

    # dealing with non-exist users & business
    both_new_data = full_data.filter(lambda x: x[1][1] is None).map(lambda x: (x[0], x[1][0])). \
        filter(lambda x: x[1] not in business)
    for pair in both_new_data.collect():  # issue those both new item and user, the global average rating
        dict_cold_start[pair] = global_avg

    # dealing with non-exist users
    new_users_exist_business = full_data.filter(lambda x: x[1][1] is None).map(lambda x: (x[0], x[1][0])). \
        filter(lambda x: x[1] in business)
    for pair in new_users_exist_business.collect():  # issue those new users, the avg rating of the item
        dict_cold_start[pair] = business_ra[pair[1]]

    # Now when the users that exist in training set
    exist_users_full = full_data.filter(lambda x: x[1][1] is not None).map(lambda x: (x[0], x[1][0])). \
        filter(lambda x: x[1] not in business)
    for pair in exist_users_full.collect():  # issue those new businesses, the avg rating of the users
        dict_cold_start[pair] = user_ra[pair[0]]

    # Now deal with the un-interrupted data
    full_rdd = full_data.filter(lambda x: x[1][1] is not None).map(lambda x: ((x[0], x[1][0]), x[1][1])). \
        filter(lambda x: x[0][1] in business)
    # print(len(full_rdd.collect()))
    full = full_rdd.map(lambda x: prediction_func(x)).map(lambda x: ((x[0], x[1]), x[2]))
    for pair in full.collect():
        dict_cold_start[pair[0]] = pair[1]

    # print(len(dict_cold_start))
    pred = val_data.map(lambda x: (x[0], x[1], dict_cold_start[x])).collect()
    # print(pred)
    # print(mean_squared_error(true_res, pred, squared=False))
    with open(out_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for i in pred:
            writer.writerow([i[0], i[1], i[2]])


