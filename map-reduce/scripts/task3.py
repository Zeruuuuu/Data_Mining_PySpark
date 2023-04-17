from pyspark import SparkContext
import json
import sys
import time


def read_rdd(input_file):
    result_rdd = sc.textFile(input_file).map(lambda row: json.loads(row))
    return result_rdd


if __name__ == '__main__':
    # Read the json file into RDD and extract desired columns
    review_path = sys.argv[1]
    business_path = sys.argv[2]
    out_path_1 = sys.argv[3]
    out_path_2 = sys.argv[4]
    sc = SparkContext('local[*]', 'task3')
    review_rdd = read_rdd(review_path).map(lambda row: (row['business_id'], row['stars']))
    business_rdd = read_rdd(business_path).map(lambda row: (row['business_id'], row['city']))

    # Join the RDDs
    # Get key-value pair of city-star and the average by cities
    rdd = review_rdd.join(business_rdd).map(lambda row: (row[1][1], row[1][0])) \
        .mapValues(lambda row: (row, 1)).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))\
        .mapValues(lambda row: row[0]/row[1]).sortBy(lambda row: row[0]).sortBy(lambda row: row[1], False)

    # Prepare the file to be exported
    with open(out_path_1, 'w') as outfile:
        outfile.write("city" + "," + "stars" + "\n")
        for i in rdd.collect():
            outfile.write(str(i[0]) + ',' + str(i[1]) + "\n")

    # Part B: Compare the results of different sorting
    # Use spark
    start_spark = time.clock()
    review = read_rdd(review_path).map(lambda row: (row['business_id'], row['stars']))
    business = read_rdd(business_path).map(lambda row: (row['business_id'], row['city']))
    rdd = review.join(business).map(lambda row: (row[1][1], row[1][0])) \
        .mapValues(lambda row: (row, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda row: row[0] / row[1]).takeOrdered(10, key=lambda row: -row[1])
    print(rdd)
    end_spark = time.clock()
    runtime_spark = end_spark - start_spark

    # Use python
    start_python = time.clock()
    review = read_rdd(review_path).map(lambda row: (row['business_id'], row['stars']))
    business = read_rdd(business_path).map(lambda row: (row['business_id'], row['city']))
    rdd_py = review.join(business).map(lambda row: (row[1][1], row[1][0])) \
        .mapValues(lambda row: (row, 1)).reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda row: row[0] / row[1])
    my_list = rdd_py.collect()
    my_list.sort(key=lambda row: -row[1])
    my_list = my_list[0:10]
    print(my_list)
    end_python = time.clock()
    runtime_python = end_python - start_python

    # Generate the frame for json display
    my_dict = {'m1': runtime_python,
               'm2': runtime_spark,
               'reason': 'Spark is slightly faster than python. If we do everything including loading data,'
                         'merging the tables, calculating the averages without parallel computing in spark, doing '
                         'python will be much slower. For this sorting part, spark can partition the data and '
                         'compute the sorting task with many cores but pythons can only use one single core to '
                         'deal with it. The advantages will be more obvious if the data is large scale.'}

    # Out put as json
    with open(out_path_2, 'w') as outfile:
        json.dump(my_dict, outfile, indent=2)
