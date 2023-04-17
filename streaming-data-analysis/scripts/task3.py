import random
import sys
import csv
from blackbox import BlackBox


if __name__ == "__main__":
    bx = BlackBox()
    input_file = sys.argv[1]
    num_ask = int(sys.argv[3])
    stream_size = int(sys.argv[2])
    output_file = sys.argv[4]

    random.seed(553)
    reservoir = []
    count = 0
    record = []
    for iteration in range(num_ask):
        users = bx.ask(input_file, stream_size)
        for i, user in enumerate(users):
            if len(reservoir) < 100:
                reservoir.append(user)
            else:
                prob = 100/(stream_size*iteration + i + 1)
                if random.random() < prob:
                    reservoir[random.randint(0, 99)] = user
            count += 1
            if count % 100 == 0:
                record.append((str(count), reservoir[0], reservoir[20], reservoir[40], reservoir[60], reservoir[80]))

    with open(output_file, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['seqnum', '0_id', '20_id', '40_id', '60_id', '80_id'])
        for item in record:
            writer.writerow([item[0], item[1], item[2], item[3], item[4], item[5]])





