import numpy as np
import pandas as pd
import sys
import mykmeanssp


def print_file(filename):
    f = open(filename)
    for line in f:
        print(line[:-1])
    print()
    f.close()


def create_datafile_from_merged_inputs(filename, data):
    file = open(filename, "w")
    temp = data.reset_index()
    for index, row in temp.iterrows():
        lst_row = list(row[1:])
        cluster = [(str(dim)) for dim in lst_row]
        file.write(','.join(cluster) + "\n")
    file.close()


def create_clusters_file(filename, data):
    file = open(filename, "w")
    for lst_row in data:
        cluster = [(str(dim)) for dim in lst_row[0]]
        file.write(",".join(cluster) + "\n")
    file.close()


def submit_args():
    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print("Invalid Input! 1")
        return 1
    try:
        k = int(sys.argv[1])
        if len(sys.argv) == 6:
            max_iter = int(sys.argv[2])
            eps = float(sys.argv[3])
            file_1_name = sys.argv[4]
            file_2_name = sys.argv[5]
        else:
            max_iter = 300
            eps = float(sys.argv[2])
            file_1_name = sys.argv[3]
            file_2_name = sys.argv[4]

        if type(k) != int or type(max_iter) != int or max_iter <= 0 or k <= 0 or\
                type(eps) not in [int, float] or eps < 0.0:
            print("Invalid Input!")
            return 1

        file_1 = open(file_1_name)
        file_2 = open(file_2_name)
    except (ValueError, OSError):
        print("Invalid Input! 2")
        return 1
    return k, max_iter, eps, file_1, file_2


def extract_vec(df, num):
    num = float(num)
    choice = df[df.index == num]
    vec = choice.to_numpy()
    return vec


def euc_nor(vec):
    vec = vec**2
    return np.sum(vec)


def sort_df(file_1, file_2):
    # Read the dataFrames
    try:
        df1 = pd.read_csv(file_1, header=None)
        df2 = pd.read_csv(file_2, header=None)
        # Set the first column to 'key'
        df1.rename(columns={0: 'key'}, inplace=True)
        df2.rename(columns={0: 'key'}, inplace=True)
        # Join the dfs on 'key'
        df = df1.merge(df2, on='key', how='inner')
        df.set_index('key', inplace=True)
        # sort the indicis
        df.sort_index(inplace=True)
    except:
        print("Invalid Input! 3")
        return 1
    return df


def kmeans_pp(k, max_iter, eps, df):
    i = 1
    np.random.seed(0)
    create_datafile_from_merged_inputs("merged_input.txt", df)
    # Choose the index randomly and set it to a numpy array
    rows = df.index

    if k > len(rows):
        print("invalid Input! 4")
        return 1

    rnd_num = np.random.choice(rows, 1)
    rnd_vec = extract_vec(df, rnd_num)
    means = [rnd_vec]
    means_indices = [rnd_num]
    n = len(df.index)
    while i != k:
        D_lst = [0 for m in range(n)]
        P_lst = [0 for m in range(n)]
        # Create the d list containing all the Dl
        for l in range(n):
            min = 100000000000000
            vec = extract_vec(df, l)
            for j in range(i):
                cur_norm = euc_nor(vec - means[j])
                if cur_norm < min:
                    min = cur_norm
            D_lst[l] = min
        # Calculate the probabilities
        for l in range(n):
            P_lst[l] = D_lst[l] / sum(D_lst)
        i += 1
        rnd_num = np.random.choice(rows, 1, p=P_lst)
        rnd_vec = extract_vec(df, rnd_num)
        means.append(rnd_vec)
        means_indices.append(rnd_num)
    create_clusters_file("cluster_file.txt", means)
    return_lst = [str(i)[1:-2] for i in means_indices]
    return_str = ''
    for i in return_lst:
        return_str += str(i) + ","
    print(return_str[:-1])
    return 0


def main():
    args = submit_args()
    cluster_filename = "cluster_file.txt"
    data_filename = "merged_input.txt"
    if args == 1:
        return 1
    k, max_iter, eps, file_1, file_2 = args
    df = sort_df(file_1, file_2)
    file_1.close()
    file_2.close()
    if kmeans_pp(k, max_iter, eps, df) == 1:
        return 1
    try:
        kmeans_success = mykmeanssp.k_means(k, max_iter, eps, data_filename, cluster_filename)
    except:
        print("An Error Has Occurred")
        return 1
    if kmeans_success == 0:
        print_file(cluster_filename)
        return 0
    else:
        return 1


if __name__ == '__main__':
    main()



