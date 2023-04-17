import os
import sys
from graphframes import GraphFrame
from pyspark import SparkContext
from pyspark.sql import SQLContext

os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"


def get_graph(ls):
    for pair1 in ls:
        for pair2 in ls:
            if (pair1[0] != pair2[0]) and (len(set(pair1[1]).intersection(set(pair2[1]))) >= my_filter):
                points.update([(pair1[0],), (pair2[0],)])
                edges.add((pair1[0], pair2[0]))
    return points, edges


if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('WARN')

    my_filter = float(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    # Create the rdd that group by user id
    my_rdd = sc.textFile(input_path).map(lambda x: x.split(','))
    header = my_rdd.first()
    user_to_business_list = my_rdd.filter(lambda x: x != header).map(lambda x: (x[0], x[1])) \
        .groupByKey().mapValues(set).mapValues(list).collect()

    points, edges = set(), set()
    points, edges = get_graph(user_to_business_list)
    points, edges = list(points), list(edges)
    # Initialize SQL context to create dataframe
    sql_context = SQLContext(sc)
    points = sql_context.createDataFrame(points, ['id'])
    edges = sql_context.createDataFrame(edges, ['src', 'dst'])
    # We got the points and edges we need. Time to build the graph
    my_graph = GraphFrame(points, edges)
    community = my_graph.labelPropagation(maxIter=5)
    community_rdd = community.rdd

    my_com = community_rdd.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: x[1]).map(list).\
        map(lambda x: sorted(x)) \
        .sortBy(lambda x: (len(x), sorted(x))).collect()

    with open(output_path, 'w') as outfile:
        for i in my_com:
            i = sorted(i)
            outfile.write(', '.join("'"+str(j)+"'" for j in i) + '\n')