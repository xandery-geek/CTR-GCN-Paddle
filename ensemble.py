import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
import csv
from util.util import print_color


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase',
                        default='predict',
                        required=False,
                        choices={"test", "predict"},
                        help="the phase of ensemble")

    parser.add_argument('--dataset',
                        required=False,
                        default='fsd',
                        choices={'fsd'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    arg = parser.parse_args()
    return arg


def load_score(arg):
    """

    :param arg: arg
    :return: r1, r2, r3, r4 (joint, bone, joint_motion, bone_motion)
    """
    r1, r2, r3, r4 = None, None, None, None

    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r3:
            r3 = list(pickle.load(r3).items())
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as r4:
            r4 = list(pickle.load(r4).items())

    return r1, r2, r3, r4


def test(arg):
    dataset = arg.dataset
    if dataset == 'fsd':
        data_path = "data/fsd/raw_data/train_data.npy"
        npz_data = np.load(data_path)
        label = np.zeros(npz_data.shape[0], dtype=np.int64)
    else:
        raise NotImplementedError

    r1, r2, r3, r4 = load_score(arg)
    right_num = total_num = right_num_5 = 0

    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        arg.alpha = [0.6, 0.6, 0.4, 0.4]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))


def predict(arg):
    r1, r2, r3, r4 = load_score(arg)

    prediction = []
    if arg.joint_motion_dir is not None and arg.bone_motion_dir is not None:
        print_color("Ensemble with all information")
        arg.alpha = [0.6, 0.6, 0.4, 0.4]
        for i in range(len(r1)):
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            r = np.argmax(r)
            prediction.append(r)

    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        print_color("Ensemble with joint, bone and joint motion information")
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(r1))):
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            r = np.argmax(r)
            prediction.append(r)
    else:
        print_color("Ensemble with joint and bone information")
        for i in tqdm(range(len(r1))):
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 + r22 * arg.alpha
            r = np.argmax(r)
            prediction.append(r)

    with open('prediction.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "predict_category"])
        writer.writerows(zip(range(len(prediction)), prediction))


if __name__ == "__main__":
    arg = parser_arg()
    if arg.phase == "test":
        test(arg)
    elif arg.phase == "predict":
        predict(arg)
    else:
        print("not {}".format(arg.phase))
