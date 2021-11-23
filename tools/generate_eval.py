import os
import pickle
import json
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def load_train_data():
    # N, C, T, V, M
    train_data = np.load('../data/fsd/raw_data/train_data.npy')
    train_label = np.load('../data/fsd/raw_data/train_label.npy')
    return train_data, train_label


def load_test_data():
    test_data = np.load("../data/fsd/raw_data/test_A_data.npy")
    return test_data


def load_score():
    work_dir = {"joint_dir": "fsd_joint",
                "bone_dir": "fsd_bone",
                "joint_motion_dir": "fsd_jmotion",
                "bone_motion_dir": "fsd_bmotion"}

    for key in work_dir:
        work_dir[key] = '../work_dir/' + work_dir[key]

    with open(os.path.join(work_dir["joint_dir"], 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(work_dir["bone_dir"], 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(os.path.join(work_dir["joint_motion_dir"], 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(work_dir["bone_motion_dir"], 'epoch1_test_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())
    return r1, r2, r3, r4


def count_dataset(label, index):
    """
    count number for each class in dataset
    :param label:
    :param index:
    :return:
    """
    count = []
    for i in range(30):
        count.append(0)

    for i in index:
        count[label[i]] += 1

    inverse_class_freq(count)
    for i, c in enumerate(count):
        print("{}:{}".format(i, c))


def analyse_data():
    r1, r2, r3, r4 = load_score()

    class_count = [0] * 30
    identical_class = [30] * len(r1)
    for i in range(len(r1)):
        c1 = np.argmax(r1[i][1])
        c2 = np.argmax(r2[i][1])
        c3 = np.argmax(r3[i][1])
        c4 = np.argmax(r4[i][1])

        if c1 == c2 == c3 == c4:
            identical_class[i] = c1
            class_count[c1] += 1
        else:
            c0 = np.argmin(r1[i][1])
            identical_class[i] = c0

    for i, count in enumerate(class_count):
        print("{}: {}".format(i, count))

    # print("identical ratio: {}".format(float(len(identical_class))/len(r1)))
    #
    # with open('identical_class.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['sample_index', 'predict_category'])
    #     writer.writerows(zip(range(len(identical_class)), identical_class))


def generate_eval_dataset():
    r1, r2, r3, r4 = load_score()

    test_data = load_test_data()
    test_index = []
    class_count = [0] * 30
    for i in range(len(r1)):
        c1 = np.argmax(r1[i][1])
        c2 = np.argmax(r2[i][1])
        c3 = np.argmax(r3[i][1])
        c4 = np.argmax(r4[i][1])

        if c1 == c2 == c3 == c4:
            test_index.append([i, c1])
            class_count[c1] += 1

    test_index = np.array(test_index)
    eval_data = test_data[test_index[:, 0]]
    eval_label = test_index[:, 1]

    # Calculate the number of samples to be obtained from the train dataset
    for i, count in enumerate(class_count):
        if count < 3:
            class_count[i] = 3 - count
        else:
            class_count[i] = 0

    print(class_count)
    train_data, train_label = load_train_data()
    train_len = train_data.shape[0]
    train_index = list(range(train_len))

    for i, count in enumerate(class_count):
        if count > 0:
            class_index = np.where(train_label == i)[0]
            class_index_len = class_index.shape[0]
            count = class_index_len * 0.5 if count > class_index_len * 0.5 else count
            class_index = class_index[:count]

            class_index = np.array(class_index)
            eval_data = np.concatenate([eval_data, train_data[class_index]])
            eval_label = np.concatenate([eval_label, train_label[class_index]])

            for index in class_index:
                train_index.remove(index)

    print("Train Dataset: {}".format(len(train_index)))
    count_dataset(train_label, train_index)
    print("Eval Dataset: {}".format(len(eval_label)))
    count_dataset(eval_label, range(len(eval_label)))

    np.save("eval_data.npy", eval_data)
    np.save("eval_label.npy", eval_label)
    np.save("train_index.npy", np.array(train_index))


def partition_train_eval():
    """
    partition dataset into train dataset and eval dataset according to K-Fold
    :return:
    """
    data, label = load_train_data()

    kf = KFold(n_splits=10, shuffle=True)
    train_index, test_index = next(kf.split(data))

    print("train dataset")
    count_dataset(label, train_index)
    print("\ntest dataset")
    count_dataset(label, test_index)

    np.save('train_index.npy', np.array(train_index))
    np.save('test_index.npy', np.array(test_index))


def inverse_class_freq(freq):
    """
    inverse class frequency for focal loss which needs alpha parameters
    :param freq:
    :return:
    """
    inverse_freq = [1/i for i in freq]
    print(inverse_freq)
    inverse_freq_sum = sum(inverse_freq)
    inverse_freq = [round(i/inverse_freq_sum, 3) for i in inverse_freq]
    print(inverse_freq)


if __name__ == '__main__':
    generate_eval_dataset()
    # analyse_data()

    # partition_train_eval()
