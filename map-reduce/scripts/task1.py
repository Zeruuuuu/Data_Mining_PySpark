from pyspark import SparkContext
import json
import sys


def get_num(rdd):
    number = rdd.map(lambda row: 1).reduce(lambda a, b: a+b)  # Map each row of data into 1, and add them up with reduce
    return number


def get_num_2018(rdd):
    number = rdd.filter(lambda row: row['date'][0:4] == '2018').map(lambda row: 1).reduce(lambda a, b: a+b)
    return number


def get_distinct_users(rdd):
    number = rdd.map(lambda row: row['user_id']).distinct().count()
    return number


def get_top_10_user(rdd):
    top = rdd.map(lambda row: (row['user_id'], 1)).reduceByKey(lambda a, b: a+b).takeOrdered(10, key=lambda row:
    (-row[1], row[0]))
    return top


def get_distinct_business(rdd):
    number = rdd.map(lambda row: row['business_id']).distinct().count()
    return number


def get_top_10_business(rdd):
    top = rdd.map(lambda row: (row['business_id'], 1)).reduceByKey(lambda a, b: a+b).takeOrdered(10, key=lambda row:
    (-row[1], row[0]))
    return top


if __name__ == '__main__':
    file_path = sys.argv[1]
    output_path = sys.argv[2]
    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('OFF')
    review_rdd = sc.textFile(file_path).map(lambda row: json.loads(row))
    my_dict = {'n_review': get_num(review_rdd), 'n_review_2018': get_num_2018(review_rdd),
               'n_user': get_distinct_users(review_rdd), 'top10_user': get_top_10_user(review_rdd),
               'n_business': get_distinct_business(review_rdd), 'top10_business': get_top_10_business(review_rdd)}
    with open(output_path, 'w') as outfile:
        json.dump(my_dict, outfile)




