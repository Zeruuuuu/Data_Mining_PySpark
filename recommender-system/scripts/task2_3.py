from xgboost import XGBRegressor
import pandas as pd
from pyspark import SparkContext
import sys
import json
from collections import defaultdict
import csv
import math


# import everything from task 2_1
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
    my_threhold = 200
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
        if len(set(business_to_user_rating[b]) &
               set(business_to_user_rating[business_pred])) >= my_threhold:
            bus_update.append(b)
    bus_final = []
    for b in bus_update:
        pearson = pearson_co(business_pred, b)
        if pearson > 0.7:
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


def noise(x):
    if x is None:
        res = 0
    elif x == 'quiet':
        res = 1
    elif x == 'average':
        res = 2
    elif x == 'loud':
        res = 3
    else:
        res = 4
    return res


def encoding(x):
    if x is None:
        res = 0
    elif x == True or x == "True":
        res = 1
    else:
        res = 0
    return res


def encoding_num(x):
    if x is None:
        res = 0
    else:
        res = float(x)
    return res


def hybrid(my_list):
    rating1 = my_list[0]
    rating2 = my_list[1]
    res = rating1 * 0.78 + rating2 * 0.22
    return res


if __name__ == "__main__":
    sc = SparkContext('local[*]', 'task2_3')
    sc.setLogLevel('WARN')

    folder_path = sys.argv[1]
    training_input_path = folder_path + '/yelp_train.csv'
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

    # import everything from xgboost
    business = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)). \
        map(lambda x: (x['business_id'], x['stars'], x['review_count'], x['is_open'], x['attributes'], x['city']))
    # print(business.take(2))
    # Get (city, avg star of city)
    city_rdd = business.map(lambda x: (x[5], x[1])).groupByKey().mapValues(lambda x: sum(x) / len(x))
    business = business.map(lambda x: (x[5], (x[0], x[1], x[2], x[3], x[4]))).leftOuterJoin(city_rdd). \
        map(lambda x: (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1]))
    review_avg = sum(business.map(lambda x: x[2]).collect()) / len(business.map(lambda x: x[2]).collect())

    # Load the user-business-rating
    header = sc.textFile(folder_path + '/yelp_train.csv').map(lambda x: x.split(',')).first()
    train_rdd = sc.textFile(folder_path + '/yelp_train.csv').map(lambda x: x.split(',')).filter(lambda x: x != header). \
        map(lambda x: (x[0], x[1], float(x[2])))  # (user, business, rating)
    user_rating = train_rdd.map(lambda x: (x[0], x[2])).groupByKey().mapValues(lambda x: sum(x) / len(x))
    business_rating = train_rdd.map(lambda x: (x[1], x[2])).groupByKey().mapValues(lambda x: sum(x) / len(x))
    train_rdd_pair = train_rdd.map(lambda x: (x[0], (x[1], x[2])))
    train_rdd_pair = train_rdd_pair.leftOuterJoin(user_rating).map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
    # above format: (business, (user, rating, avg user rating))
    train_rdd_pair = train_rdd_pair.leftOuterJoin(business_rating). \
        map(lambda x: ((x[0], x[1][0][0]), (x[1][0][1], x[1][0][2], x[1][1])))
    # print(train_rdd_pair.take(3))  # (business, user, rating, user avg, business avg)
    tips_rdd = sc.textFile(folder_path + '/tip.json').map(lambda x: json.loads(x)). \
        map(lambda x: ((x['business_id'], x['user_id']), x['likes'])).groupByKey().mapValues(sum)
    avg_tip = sum(tips_rdd.map(lambda x: x[1]).collect()) / len(tips_rdd.map(lambda x: x[1]).collect())

    train_rdd_pair = train_rdd_pair.leftOuterJoin(tips_rdd). \
        map(lambda x: (x[0][0], (x[0][1], x[1][0][0], x[1][0][1], x[1][0][2], x[1][1]
    if x[1][1] is not None else avg_tip)))

    attributes = business.map(lambda x: (x[0], x[4])).filter(lambda x: x[1] is not None). \
        map(lambda x: (x[0], x[1]['NoiseLevel'] if 'NoiseLevel' in x[1] else None,
                       x[1]['RestaurantsPriceRange2'] if 'RestaurantsPriceRange2' in x[1] else None,
                       x[1]['GoodForKids'] if 'GoodForKids' in x[1] else None,
                       x[1]['RestaurantsGoodForGroups'] if 'RestaurantsGoodForGroups' in x[1] else None,
                       x[1]['BikeParking'] if 'BikeParking' in x[1] else None,
                       x[1]['RestaurantsTableService'] if 'RestaurantsTableService' in x[1] else None))

    attributes = attributes.map(lambda x: (x[0], (noise(x[1]), encoding_num(x[2]), encoding(x[3]),
                                                  encoding(x[4]), encoding(x[5]), encoding(x[6]))))
    avg_noise = sum(attributes.map(lambda x: x[1][0]).collect()) / len(attributes.map(lambda x: x[1][0]).collect())
    # print(attributes.take(5))
    feature_space = train_rdd_pair.leftOuterJoin(attributes). \
        map(lambda x: (
        x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4],
               x[1][1][0] if (x[1][1] is not None) else avg_noise
               , x[1][1][1] if (x[1][1] is not None) else 1.5
               , x[1][1][2] if (x[1][1] is not None) else 0
               , x[1][1][3] if (x[1][1] is not None) else 0
               , x[1][1][4] if (x[1][1] is not None) else 1
               , x[1][1][5] if (x[1][1] is not None) else 0)))
    business_checkin = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)). \
        map(lambda x: (x['business_id'], sum(x['time'].values()))).groupByKey().mapValues(lambda x: sum(x))
    avg_checkin = sum(business_checkin.map(lambda x: x[1]).collect()) / len(
        business_checkin.map(lambda x: x[1]).collect())
    # Business_id, total number of check-in
    feature_space = feature_space.leftOuterJoin(business_checkin). \
        map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                              x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10],
                              x[1][1] / avg_checkin if x[1][1] is not None else 0)))
    # print(feature_space.collect())
    # Format: (business_id, (user, rating, avg_user, avg_bus, noise level, price level, good for kid, good for group))

    business_rdd = business.map(lambda x: (x[0], (x[1], x[2], x[3], x[5])))
    star_avg = sum(business_rdd.map(lambda x: x[1][0]).collect()) / len(business_rdd.map(lambda x: x[1][0]).collect())
    open_avg = sum(business_rdd.map(lambda x: x[1][2]).collect()) / len(business_rdd.map(lambda x: x[1][2]).collect())
    city_avg = sum(business_rdd.map(lambda x: x[1][3]).collect()) / len(business_rdd.map(lambda x: x[1][3]).collect())
    # print(business_rdd.take(2))
    feature_space = feature_space.leftOuterJoin(business_rdd). \
        map(lambda x: (x[1][0][0], (x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                       x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11],
                       x[1][1][0] if (x[1][1] is not None) else star_avg,
                       x[1][1][1] / review_avg if (x[1][1] is not None) else 0.5,
                       x[1][1][2] if (x[1][1] is not None) else open_avg,
                       x[1][1][3] if (x[1][1] is not None) else city_avg)))

    user_data = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)). \
        map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], x['useful'], x['fans'])))
    avg_rev = sum(user_data.map(lambda x: x[1][0]).collect()) / len(user_data.map(lambda x: x[1][0]).collect())
    avg_star = sum(user_data.map(lambda x: x[1][1]).collect()) / len(user_data.map(lambda x: x[1][1]).collect())
    avg_useful = sum(user_data.map(lambda x: x[1][2]).collect()) / len(user_data.map(lambda x: x[1][2]).collect())
    avg_fans = sum(user_data.map(lambda x: x[1][3]).collect()) / len(user_data.map(lambda x: x[1][3]).collect())
    feature_space = feature_space.leftOuterJoin(user_data). \
        map(lambda x: (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6],
                       x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13],
                       x[1][0][14], x[1][1][0] / avg_rev if (x[1][1] is not None) else 0,
                       x[1][1][1] if (x[1][1] is not None) else avg_star,
                       x[1][1][2] if (x[1][1] is not None) else avg_useful,
                       x[1][1][3] if (x[1][1] is not None) else avg_fans))

    pd.set_option('display.max_columns', 1000)

    feature_df = pd.DataFrame(feature_space.collect())
    # print(feature_df[0])
    train_y = feature_df[0]
    training_data = feature_df.drop(columns=[0, 1, 2])
    training_data.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # avg_user, avg_business, noise level, PR, GFK, GFG, check-in number, business_star, review_number, is_open,
    # city_rating_average
    # print(training_data.head())

    # prepare validation data
    val_header = sc.textFile(val_path).map(lambda x: x.split(',')).first()
    val_rdd = sc.textFile(val_path).map(lambda x: x.split(',')).filter(lambda x: x != val_header). \
        map(lambda x: (x[0], x[1]))  # (user, business)
    val_data = val_rdd.map(lambda x: (x[0], x[1]))
    val_rdd_pair = val_data.leftOuterJoin(user_rating). \
        map(lambda x: (x[1][0], (x[0], x[1][1] if x[1][1] is not None else 3)))
    # above format: (business, (user, avg user rating))
    val_rdd_pair = val_rdd_pair.leftOuterJoin(business_rating). \
        map(lambda x: ((x[0], x[1][0][0]), (x[1][0][1], x[1][1] if x[1][1] is not None else 3)))
    # print(val_rdd_pair.take(3))  # (business, user, user avg, business avg)

    val_rdd_pair = val_rdd_pair.leftOuterJoin(tips_rdd). \
        map(lambda x: (x[0][0], (x[0][1], x[1][0][0], x[1][0][1], x[1][1] if x[1][1] is not None else avg_tip)))

    val_feature = val_rdd_pair.leftOuterJoin(attributes). \
        map(lambda x: (
        x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3],
               x[1][1][0] if (x[1][1] is not None) else avg_noise
               , x[1][1][1] if (x[1][1] is not None) else 1.5
               , x[1][1][2] if (x[1][1] is not None) else 0
               , x[1][1][3] if (x[1][1] is not None) else 0
               , x[1][1][4] if (x[1][1] is not None) else 1
               , x[1][1][5] if (x[1][1] is not None) else 0)))

    val_feature = val_feature.leftOuterJoin(business_checkin). \
        map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                              x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9],
                              x[1][1] / avg_checkin if x[1][1] is not None else 0)))

    val_feature = val_feature.leftOuterJoin(business_rdd). \
        map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                       x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10],
                       x[1][1][0] if (x[1][1] is not None) else star_avg,
                       x[1][1][1] / review_avg if (x[1][1] is not None) else 0.5,
                       x[1][1][2] if (x[1][1] is not None) else open_avg,
                       x[1][1][3] if (x[1][1] is not None) else city_avg)))

    val_feature = val_feature.leftOuterJoin(user_data). \
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6],
                       x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13],
                       x[1][0][14], x[1][1][0] / avg_rev if (x[1][1] is not None) else 0,
                       x[1][1][1] if (x[1][1] is not None) else avg_star,
                       x[1][1][2] if (x[1][1] is not None) else avg_useful,
                       x[1][1][3] if (x[1][1] is not None) else avg_fans))

    val_df = pd.DataFrame(val_feature.collect())
    user_business_ids = val_df[[0, 1]]  # user, business
    # print(user_business_ids)
    val_features = val_df.drop(columns=[0, 1, 2, 3])
    # avg_user, avg_business, tip, noise level, PR, GFK, GFG, check-in number, business_star, review_number, is_open,
    # city_rating_average
    val_features.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # print(val_features.head())

    # Model
    xgb = XGBRegressor(learning_rate=0.2)
    xgb.fit(training_data, train_y)
    pred_xgb = xgb.predict(val_features)
    dictionary = defaultdict(list)
    for i, row in user_business_ids.iterrows():
        dictionary[tuple(row)].append(pred_xgb[i])

    for t in pred:
        dictionary[(t[0], t[1])].append(t[2])

    # Output
    val_rdd = sc.textFile(val_path).map(lambda x: x.split(',')).filter(lambda x: x != val_header). \
        map(lambda x: (x[0], x[1]))
    predict = val_rdd.map(lambda x: (x[0], x[1], hybrid(dictionary[x]))).collect()
    with open(out_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for row in predict:
            writer.writerow([row[0], row[1], row[2]])



