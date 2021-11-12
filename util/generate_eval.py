import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt


def load_score(work_dir):
    r1, r2, r3, r4 = None, None, None, None
    with open(os.path.join(work_dir["joint_dir"], 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(work_dir["bone_dir"], 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(os.path.join(work_dir["joint_motion_dir"], 'epoch1_test_score.pkl'), 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(os.path.join(work_dir["bone_motion_dir"], 'epoch1_test_score.pkl'), 'rb') as r4:
        r4 = list(pickle.load(r4).items())
    return r1, r2, r3, r4


def normalize(x):
    x_min = np.min(x)
    x = [i-x_min for i in x]
    x_sum = np.sum(x)
    x = x / x_sum
    return x


def normalize2(x):
    x_min = np.min(x)
    x_max = np.max(x)
    x = (x - x_min) / (x_max - x_min)
    return x


def analyse_data():
    work_dir = {"joint_dir": "fsd_joint",
                "bone_dir": "fsd_bone",
                "joint_motion_dir": "fsd_jmotion",
                "bone_motion_dir": "fsd_bmotion"}

    for key in work_dir:
        work_dir[key] = '../work_dir/' + work_dir[key]

    alpha = [0.6, 0.6, 0.4, 0.4]
    r1, r2, r3, r4 = load_score(work_dir)
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


if __name__ == '__main__':
    generate_eval_dataset()
    # analyse_data()
