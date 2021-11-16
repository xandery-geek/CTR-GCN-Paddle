import numpy as np

from paddle.io import Dataset
from util.util import print_color
# from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, phase='train', p_interval=1, split='train', random_choose=False,
                 random_shift=False, random_move=False, random_rot=False, window_size=-1, normalization=False,
                 debug=False, use_mmap=False, bone=False, motion=False, augmentation=None):
        """
        :param data_path:
        :param label_path:
        :param phase:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param motion: use motion modality or not, if bone is False, return the motion of joints, else return the motion of bones
        :param augmentation: data augmentation
        """

        super().__init__()
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.motion = motion
        self.augmentation = augmentation
        self.phase = phase
        self.data = None
        self.label = None
        self.sample_name = None

        print_color("\n>>> Loading {} dataset <<<".format(self.split))
        if self.bone:
            print_color(">>> Using bone information! <<<")
        else:
            print_color(">>> Using joint information! <<<")

        if self.motion:
            print_color(">>> Using motion information! <<<")
        else:
            print_color(">>> Not using motion information! <<<")

        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # data: N C V T M
        if self.phase == "eval":
            npz_data = np.load(self.data_path)
            npz_label = np.load(self.label_path)

            if self.split == "train":
                print_color(">>> Loading train index for evaluation <<<")
                train_index = np.load('data/fsd/raw_data/train_index.npy')
                self.data = npz_data[train_index]
                self.label = npz_label[train_index]

            elif self.split == "test":
                # print_color(">>> Loading test index for evaluation <<<")
                # test_index = np.load('data/fsd/test_index.npy')
                # self.data = npz_data[test_index]
                # self.label = npz_label[test_index]
                self.data = npz_data
                self.label = npz_label
        else:
            print_color(">>> Loading data from {} <<<".format(self.data_path))
            npz_data = np.load(self.data_path)
            if self.label_path:
                print_color(">>> Loading label from {} <<<".format(self.label_path))
                npz_label = np.load(self.label_path)
            else:
                npz_label = np.zeros(npz_data.shape[0], dtype=np.int64)
            self.data = npz_data
            self.label = npz_label

        del npz_data
        del npz_label
        if self.augmentation:
            self.data_argumentation()

        # set sample name for each sample
        self.sample_name = [self.split + '_' + str(i) for i in range(len(self.data))]

    def data_argumentation(self):
        if self.augmentation == "avg":
            "calculate average of neighbor frame"
            print_color(">>> Augmenting dataset with 'avg' <<<")
            N, C, T, V, M = self.data.shape
            new_data = np.zeros(shape=self.data.shape, dtype=self.data.dtype)
            for i in range(T-1):
                new_data[:, :, i, :, :] = 0.5 * self.data[:, :, i, :, :] + 0.5 * self.data[:, :, i+1, :, :]

            new_data[:, :, -1, :, :] = self.data[:, :, -1, :, :]
            self.data = np.concatenate((self.data, new_data), axis=0)
            self.label = np.concatenate((self.label, self.label), axis=0)

        elif self.augmentation == 'conf':
            "calculate weighted sum of neighbor frame based on confidence"
            print_color(">>> Augmenting dataset with 'confidence' <<<")
            N, C, T, V, M = self.data.shape
            new_data = np.zeros(shape=self.data.shape, dtype=self.data.dtype)
            for i in range(T-1):
                conf1 = self.data[:, 2, i, :, :]
                conf2 = self.data[:, 2, i+1, :, :]

                conf_sum = conf1 + conf2
                conf_sum = np.where(conf_sum > 0, conf_sum, 1)

                conf1 = conf1/conf_sum
                conf2 = conf2/conf_sum

                new_data[:, 0, i, :, :] = conf1 * self.data[:, 0, i, :, :] + conf2 * self.data[:, 0, i+1, :, :]
                new_data[:, 1, i, :, :] = conf1 * self.data[:, 1, i, :, :] + conf2 * self.data[:, 1, i+1, :, :]
                new_data[:, 2, i, :, :] = 0.5 * self.data[:, 2, i, :, :] + 0.5 * self.data[:, 2, i+1, :, :]

            new_data[:, :, -1, :, :] = self.data[:, :, -1, :, :]
            self.data = np.concatenate((self.data, new_data), axis=0)
            self.label = np.concatenate((self.label, self.label), axis=0)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        # data_numpy: C V T M
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)

        # if self.random_rot:
        #     data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            # bone information = \Delta(neighbor joint feature) at each frame
            from feeders.bone_pairs import fsd_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in fsd_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.motion:
            # motion information = \Delta(joint feature) at succession frame
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)
