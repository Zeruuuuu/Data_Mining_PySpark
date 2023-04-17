from xgboost import XGBRegressor
import pandas as pd
from pyspark import SparkContext
import sys
import json
from collections import defaultdict
import csv


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


if __name__ == "__main__":
    sc = SparkContext('local[*]', 'task2_2')
    folder_path = sys.argv[1]
    val_path = sys.argv[2]
    out_path = sys.argv[3]
    business = sc.textFile(folder_path + '/business.json').map(lambda x: json.loads(x)). \
        map(lambda x: (x['business_id'], x['stars'], x['review_count'], x['is_open'], x['attributes'], x['city']))
    # print(business.take(2))
    # Get (city, avg star of city)
    city_rdd = business.map(lambda x: (x[5], x[1])).groupByKey().mapValues(lambda x: sum(x) / len(x))
    business = business.map(lambda x: (x[5], (x[0], x[1], x[2], x[3], x[4]))).leftOuterJoin(city_rdd). \
        map(lambda x: (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][1]))
    review_avg = sum(business.map(lambda x : x[2]).collect())/len(business.map(lambda x : x[2]).collect())

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
    avg_tip = sum(tips_rdd.map(lambda x: x[1]).collect())/len(tips_rdd.map(lambda x: x[1]).collect())

    train_rdd_pair = train_rdd_pair.leftOuterJoin(tips_rdd).\
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
               x[1][1][0] if x[1][1] is not None else avg_noise
               , x[1][1][1] if x[1][1] is not None else 1.5
               , x[1][1][2] if x[1][1] is not None else 0
               , x[1][1][3] if x[1][1] is not None else 0
               , x[1][1][4] if x[1][1] is not None else 1
               , x[1][1][5] if x[1][1] is not None else 0)))
    business_checkin = sc.textFile(folder_path + '/checkin.json').map(lambda x: json.loads(x)).\
        map(lambda x: (x['business_id'], sum(x['time'].values()))).groupByKey().mapValues(lambda x: sum(x))
    avg_checkin = sum(business_checkin.map(lambda x: x[1]).collect())/len(business_checkin.map(lambda x: x[1]).collect())
    # Business_id, total number of check-in
    feature_space = feature_space.leftOuterJoin(business_checkin).\
        map(lambda x: (x[0], (x[1][0][0],x[1][0][1],x[1][0][2],x[1][0][3],x[1][0][4],x[1][0][5],
                              x[1][0][6],x[1][0][7],x[1][0][8], x[1][0][9], x[1][0][10],
                              x[1][1]/avg_checkin if x[1][1] is not None else 0)))
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
                       x[1][1][0] if x[1][1] is not None else star_avg,
                       x[1][1][1]/review_avg if x[1][1] is not None else 0,
                       x[1][1][2] if x[1][1] is not None else open_avg,
                       x[1][1][3] if x[1][1] is not None else city_avg)))
    pd.set_option('display.max_columns', 1000)
    user_data = sc.textFile(folder_path + '/user.json').map(lambda x: json.loads(x)). \
        map(lambda x: (x['user_id'], (x['review_count'], x['average_stars'], x['useful'], x['fans'])))
    avg_rev = sum(user_data.map(lambda x: x[1][0]).collect())/len(user_data.map(lambda x: x[1][0]).collect())
    avg_star = sum(user_data.map(lambda x: x[1][1]).collect()) / len(user_data.map(lambda x: x[1][1]).collect())
    avg_useful = sum(user_data.map(lambda x: x[1][2]).collect()) / len(user_data.map(lambda x: x[1][2]).collect())
    avg_fans = sum(user_data.map(lambda x: x[1][3]).collect()) / len(user_data.map(lambda x: x[1][3]).collect())
    feature_space = feature_space.leftOuterJoin(user_data).\
        map(lambda x: (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6],
                       x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13],
                       x[1][0][14], x[1][1][0]/avg_rev if (x[1][1] is not None) else 0,
                       x[1][1][1] if (x[1][1] is not None) else avg_star,
                       x[1][1][2] if (x[1][1] is not None) else avg_useful,
                       x[1][1][3] if (x[1][1] is not None) else avg_fans))

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
    val_rdd_pair = val_data.leftOuterJoin(user_rating).\
        map(lambda x: (x[1][0], (x[0], x[1][1] if x[1][1] is not None else 3)))
    # above format: (business, (user, avg user rating))
    val_rdd_pair = val_rdd_pair.leftOuterJoin(business_rating). \
        map(lambda x: ((x[0], x[1][0][0]), (x[1][0][1], x[1][1] if x[1][1] is not None else 3)))
    # print(val_rdd_pair.take(3))  # (business, user, user avg, business avg)

    val_rdd_pair = val_rdd_pair.leftOuterJoin(tips_rdd). \
        map(lambda x: (x[0][0], (x[0][1], x[1][0][0], x[1][0][1], x[1][1] if x[1][1] is not None else avg_tip)))

    val_feature = val_rdd_pair.leftOuterJoin(attributes). \
        map(lambda x: (
        x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][1][0] if x[1][1] is not None else avg_noise
               , x[1][1][1] if x[1][1] is not None else 1.5
               , x[1][1][2] if x[1][1] is not None else 0
               , x[1][1][3] if x[1][1] is not None else 0
               , x[1][1][4] if x[1][1] is not None else 1
               , x[1][1][5] if x[1][1] is not None else 0)))

    val_feature = val_feature.leftOuterJoin(business_checkin). \
        map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                              x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9],
                              x[1][1]/avg_checkin if x[1][1] is not None else 0)))

    val_feature = val_feature.leftOuterJoin(business_rdd). \
        map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5],
                       x[1][0][6], x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10],
                       x[1][1][0] if x[1][1] is not None else star_avg,
                       x[1][1][1]/review_avg if x[1][1] is not None else 0,
                       x[1][1][2] if x[1][1] is not None else open_avg,
                       x[1][1][3] if x[1][1] is not None else city_avg)))

    val_feature = val_feature.leftOuterJoin(user_data). \
        map(lambda x: (x[0], x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], x[1][0][4], x[1][0][5], x[1][0][6],
                       x[1][0][7], x[1][0][8], x[1][0][9], x[1][0][10], x[1][0][11], x[1][0][12], x[1][0][13],
                       x[1][0][14], x[1][1][0]/avg_rev if (x[1][1] is not None) else 0,
                       x[1][1][1] if (x[1][1] is not None) else avg_star,
                       x[1][1][2] if (x[1][1] is not None) else avg_useful,
                       x[1][1][3] if (x[1][1] is not None) else avg_fans))
    val_df = pd.DataFrame(val_feature.collect())
    user_business_ids = val_df[[0, 1]]  # business, user !!!
    # print(user_business_ids)
    val_features = val_df.drop(columns=[0, 1, 2, 3])
    # avg_user, avg_business, tip, noise level, PR, GFK, GFG, check-in number, business_star, review_number, is_open,
    # city_rating_average
    val_features.columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # print(val_features.head())

    # Model
    xgb = XGBRegressor(learning_rate=0.2)
    xgb.fit(training_data, train_y)
    pred = xgb.predict(val_features)
    dictionary = defaultdict(float)
    for i, row in user_business_ids.iterrows():
        dictionary[tuple(row)] = pred[i]
    # print(dictionary)
    with open(out_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['user_id', ' business_id', ' prediction'])
        for key, value in dictionary.items():
            writer.writerow([key[0], key[1], value])







