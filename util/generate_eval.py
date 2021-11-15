import os
import pickle
import json
import numpy as np
import csv
import matplotlib.pyplot as plt


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

    for i, c in enumerate(count):
        print("{}:{}".format(i, c))


def normalize(x):
    x_min = np.min(x)
    x = [i - x_min for i in x]
    x_sum = np.sum(x)
    x = x / x_sum
    return x


def normalize2(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


def analyse_data():
    alpha = [0.6, 0.6, 0.4, 0.4]
    r1, r2, r3, r4 = load_score()
    length = len(r1)
    ret = []
    for i in range(length):
        _, r11 = r1[i]
        _, r22 = r2[i]
        _, r33 = r3[i]
        _, r44 = r4[i]

        # r11 = normalize2(r11)
        # r22 = normalize2(r22)
        # r33 = normalize2(r33)
        # r44 = normalize2(r44)

        r = r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]
        r = normalize(r)
        r = [i.item() for i in r]
        ret.append(r)

    with open('./origin_data.json', 'w') as f:
        json.dump({"data": ret}, f)


def generate_eval_dataset():
    """
    generate eval dataset according to maximum prediction
    :return:
    """
    with open('./origin_data.json', 'r') as f:
        data = json.load(f)
    data = data["data"]
    max_data = []
    for d in data:
        max_data.append(max(d))
    # plt.plot(list(range(len(max_data))), max_data)
    # plt.show()

    sorted_max_data = sorted(max_data)
    threshold = sorted_max_data[int(len(sorted_max_data) * 0.3)]

    eval_index = []
    for i, d in enumerate(max_data):
        if d > threshold:
            eval_index.append([i, np.argmax(np.array(data[i]))])

    eval_index = np.array(eval_index)
    test_data = np.load("../data/fsd/raw_data/test_A_data.npy")
    eval_data = test_data[eval_index[:, 0]]
    eval_label = eval_index[:, 1]

    np.save("../data/fsd/raw_data/eval_data.npy", eval_data)
    np.save("../data/fsd/raw_data/eval_label.npy", eval_label)


def analyse_data2():
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


def generate_eval_dataset2():
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
        if count < 8:
            class_count[i] = 8 - count
        else:
            class_count[i] = 3

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

    np.save("../data/fsd/raw_data/eval_data.npy", eval_data)
    np.save("../data/fsd/raw_data/eval_label.npy", eval_label)
    np.save("../data/fsd/raw_data/train_index.npy", np.array(train_index))


if __name__ == '__main__':
    generate_eval_dataset2()
    # analyse_data2()
