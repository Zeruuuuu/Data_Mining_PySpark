import random
import sys
from pyspark import SparkContext
from collections import defaultdict


def get_graph(ls):
    for pair1 in ls:
        for pair2 in ls:
            if (pair1[0] != pair2[0]) and (len(set(pair1[1]).intersection(set(pair2[1]))) >= my_filter):
                points.update([pair1[0], pair2[0]])
                edges.add((pair1[0], pair2[0]))
    return points, edges


def bfs(x):
    structure = {0: x}
    current_layer_nodes = near_set[x]  # The first level of the structure
    processed_nodes = {x}
    parent_dict_lookup = defaultdict(set)
    child_dict_lookup = defaultdict(set)
    child_dict_lookup[x] = near_set[x]
    shortest_path_count = {x: 1}
    current_state = 1
    # Start getting parent & Child
    while current_layer_nodes:
        next_level = set()
        structure[current_state] = current_layer_nodes
        processed_nodes = processed_nodes | current_layer_nodes
        for node in current_layer_nodes:
            child_dict_lookup[node] = near_set[node] - processed_nodes
            for parent, children in child_dict_lookup.items():
                for child in children:
                    parent_dict_lookup[child].add(parent)
            if parent_dict_lookup[node] is not None:
                num = 0
                for i in parent_dict_lookup[node]:
                    num += shortest_path_count[i]
                shortest_path_count[node] = num
            else:
                shortest_path_count[node] = 1
            next_level = next_level | near_set[node]
        current_layer_nodes = next_level - processed_nodes
        current_state += 1
    return (x, current_state, structure, parent_dict_lookup, shortest_path_count)


def get_betweenness(x):
    bet_dict = {}
    my_node, current_state, structure, parent_dict_lookup, shortest_path_count = x[0], x[1], x[2], x[3], x[4]
    bet = defaultdict(float)
    for x in points:
        if x != my_node:
            bet[x] = 1
    while current_state > 1:
        idx = current_state - 1
        for i in structure[idx]:
            for j in parent_dict_lookup[i]:
                edge = tuple(sorted((i, j)))
                bet_dict[edge] = (shortest_path_count[j] / shortest_path_count[i]) * bet[i]  # Only part of the score
                bet[j] += (shortest_path_count[j] / shortest_path_count[i]) * bet[i]
        current_state -= 1
    my_list = []
    for k, v in bet_dict.items():
        my_list.append((k, v))
    return my_list


def get_community(node):
    community = set()
    processed = set()
    community.add(node)
    processed.add(node)
    my_queue = [node]
    while my_queue:
        current_node = my_queue.pop(0)
        for n in near_set[current_node]:
            if n not in processed:
                community.add(n)
                processed.add(n)
                my_queue.append(n)
    if len(community) == 0:
        community.add(node)
    c = list()
    processed_nodes = community
    c.append(processed_nodes)
    while True:
        node1 = random.sample(points - processed_nodes, 1)[0]
        community_sub = set()
        processed_sub = set()
        community_sub.add(node1)
        processed_sub.add(node1)
        my_queue_sub = [node1]
        while my_queue_sub:
            current_node = my_queue_sub.pop(0)
            for n in near_set[current_node]:
                if n not in processed_sub:
                    community_sub.add(n)
                    processed_sub.add(n)
                    my_queue_sub.append(n)
        if len(community) == 0:
            community_sub.add(node1)
        c.append(community_sub)
        processed_nodes = processed_nodes | community_sub
        if not points - processed_nodes:
            break
    return c


def modularity(com):
    my_modularity = 0
    for community in com:
        for i in community:
            for j in community:
                ki_kj = length[i] * length[j]
                my_modularity += my_matrix[(i, j)] - ki_kj / (2 * len(edges) / 2)
    return my_modularity / (2 * len(edges) / 2)


if __name__ == '__main__':

    sc = SparkContext('local[*]', 'task1')
    sc.setLogLevel('WARN')

    my_filter = float(sys.argv[1])
    input_path = sys.argv[2]
    output_path_betweeness = sys.argv[3]
    output_path_community = sys.argv[4]
    # Create the rdd that group by user id
    my_rdd = sc.textFile(input_path).map(lambda x: x.split(','))
    header = my_rdd.first()
    user_to_business_list = my_rdd.filter(lambda x: x != header).map(lambda x: (x[0], x[1])) \
        .groupByKey().mapValues(set).mapValues(list).collect()

    points, edges = set(), set()
    points, edges = get_graph(user_to_business_list)
    my_matrix = defaultdict(int)
    # Construct the matrix A
    for x in points:
        for y in points:
            if (x, y) in edges:
                my_matrix[(x, y)] = 1
            else:
                my_matrix[(x, y)] = 0

    near_set = defaultdict(set)
    for t in edges:
        a, b = t[0], t[1]
        near_set[a].add(b)
    nodes_rdd = sc.parallelize(points)

    b = nodes_rdd.map(lambda node: bfs(node)).map(lambda x: get_betweenness(x)).flatMap(lambda x: x) \
        .reduceByKey(lambda m, n: m + n).mapValues(lambda x: x / 2).sortBy(lambda x: (-x[1], x[0])).collect()

    with open(output_path_betweeness, 'w') as outfile:
        for row in b:
            without_tuple = str(row)[1:-1]
            outfile.write(without_tuple + '\n')

    # Community
    length = {k: len(v) for k, v in near_set.items()}
    mod = -999
    not_cut_edges = len(edges) / 2
    while not_cut_edges >= 0:
        for i, j in b:
            if j == b[0][1]:
                not_cut_edges -= 1
                near_set[i[0]].remove(i[1])
                near_set[i[1]].remove(i[0])
        random_pt = random.sample(points, 1)[0]
        my_community = get_community(random_pt)
        if modularity(my_community) > mod:
            mod = modularity(my_community)
            my_com = my_community
        b = nodes_rdd.map(lambda node: bfs(node)).map(lambda x: get_betweenness(x)).flatMap(lambda x: x) \
            .reduceByKey(lambda m, n: m + n).mapValues(lambda x: x / 2).sortBy(lambda x: (-x[1], x[0])).collect()
        if not_cut_edges <= 0:
            break
    final_community = sc.parallelize(my_com)
    com = final_community.map(lambda x: sorted(x)).sortBy(lambda x: (len(x), sorted(x))).collect()

    with open(output_path_community, 'w') as outfile:
        for row in com:
            without_tuple = str(row)[1:-1]
            outfile.write(without_tuple + '\n')
