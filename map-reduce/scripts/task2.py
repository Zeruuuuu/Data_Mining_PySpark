from pyspark import SparkContext
import json
import sys
import time


def get_top_10_business(rdd):
    top = rdd.map(lambda row: (row['business_id'], 1)).reduceByKey(lambda a, b: a+b).takeOrdered(10, key=lambda row:
    (-row[1], row[0]))
    return top


def part_func(num):
    return ord(num[0])


def function_with_customized_partition(rdd, n):
    top = rdd.map(lambda row: (row['business_id'], 1)).partitionBy(n, partitionFunc=part_func).reduceByKey(lambda a, b: a + b)\
        .takeOrdered(10, key=lambda row: (-row[1], row[0]))
    return top


if __name__ == '__main__':
    file_path = sys.argv[1]
    out_path = sys.argv[2]
    n_partition = int(sys.argv[3])
    sc = SparkContext('local[*]', 'task2')
    review_rdd = sc.textFile(file_path).map(lambda row: json.loads(row))

    # Explore the default partition function
    start_time = time.clock()
    top_rdd = get_top_10_business(review_rdd)
    end_time = time.clock()
    runtime = end_time - start_time

    # Get my own partition function
    start_time_new = time.clock()
    top_rdd_cus = function_with_customized_partition(review_rdd, n_partition)
    end_time_new = time.clock()
    runtime_new = end_time_new - start_time_new

    # Get the partition information after the mapping
    rdd_default = review_rdd.map(lambda row: (row['business_id'], 1))
    rdd_custom = review_rdd.map(lambda row: (row['business_id'], 1)).partitionBy(n_partition, partitionFunc=part_func)
    my_dict_def, my_dict_cus = {'n_partition': rdd_default.getNumPartitions(),
                                'n_items': rdd_default.glom().map(len).collect(),
                                'exe_time': runtime}, \
                               {'n_partition': rdd_custom.getNumPartitions(),
                                'n_items': rdd_custom.glom().map(len).collect(),
                                'exe_time': runtime_new}
    my_dict = {'default': my_dict_def, 'customized': my_dict_cus}

    # Dump the dictionary we got to json format
    with open(out_path, 'w') as outfile:
        json.dump(my_dict, outfile)



