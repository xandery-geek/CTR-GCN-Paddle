import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold


connection = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
              (1, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 22),
              (22, 23), (8, 12), (12, 13), (13, 14), (14, 21),
              (14, 19), (19, 20), (0, 15), (15, 17), (0, 16), (16, 18)]


def load_train_data():
    train_data = np.load('../data/fsd/raw_data/train_data.npy')
    train_label = np.load('../data/fsd/raw_data/train_label.npy')
    return train_data, train_label


def load_test_A():
    test_data = np.load('./dataset/test_A_data.npy')
    return test_data


def plot_frame(x, y, confidence=None):
    x, y = -x, -y

    # plt.axis('equal')

    plt.scatter(x, y)
    for i in range(len(x)):
        if confidence is not None:
            text = '{}:{:.2}'.format(i, confidence[i])
        else:
            text = '{}'.format(i)
        plt.text(
            x=x[i],
            y=y[i],
            s=text,
            color='r')

    for c in connection:
        x_i = [x[c[0]], x[c[1]]]
        y_i = [y[c[0]], y[c[1]]]
        plt.plot(x_i, y_i)
    plt.show()


def normalize_confidence(conf1, conf2):
    conf_sum = conf1 + conf2
    conf_sum = np.where(conf_sum > 0, conf_sum, 1)
    return conf1/conf_sum, conf2/conf_sum


def plot_skeleton(index=100, frame=30, confidence=False, augmentation=None):
    data, label = load_train_data()

    assert (0 <= index < len(data))

    print("Shape of data: {}".format(data.shape))
    print("Shape of label: {}".format(label.shape))
    # N, C, T, V, M

    for i in range(frame):
        data1 = data[index, :, i, :, :]
        data2 = data[index, :, i+1, :, :]
        data1 = np.squeeze(data1)
        data2 = np.squeeze(data2)

        print("Original")
        plot_frame(data1[0], data1[1], data1[2] if confidence else None)

        if augmentation:
            print("Augmentative")
            if augmentation == 'avg':
                data1[0] = (data1[0] + data2[0])/2
                data1[1] = (data1[1] + data2[1])/2
                data1[2] = (data1[2] + data2[2])/2
            else:
                conf1, conf2 = data1[2], data2[2]
                conf1, conf2 = normalize_confidence(conf1, conf2)
                data1[0] = conf1 * data1[0] + conf2 * data2[0]
                data1[1] = conf1 * data1[1] + conf2 * data2[1]
                data1[2] = (data1[2] + data2[2])/2
            plot_frame(data1[0], data1[1], data1[2] if confidence else None)


def count_dataset(label, index):
    count = []
    for i in range(30):
        count.append(0)

    for i in index:
        count[label[i]] += 1

    for i, c in enumerate(count):
        print("{}:{}".format(i, c))


def k_fold():
    data, label = load_train_data()

    kf = KFold(n_splits=5, shuffle=True)
    train_index, test_index = next(kf.split(data))

    count_dataset(label, train_index)
    count_dataset(label, test_index)

    np.save('train_index.npy', np.array(train_index))
    np.save('test_index.npy', np.array(test_index))


if __name__ == '__main__':
    plot_skeleton(index=1000, augmentation='conf')
