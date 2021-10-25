import matplotlib.pyplot as plt


#  [ Sat Oct 23 14:56:46 2021 ] Training epoch: 1
#  [ Sat Oct 23 15:10:38 2021 ] 	Mean training loss: 2.8872.  Mean training acc: 16.15%.
#  [ Sat Oct 23 15:10:38 2021 ] Eval epoch: 1
#  [ Sat Oct 23 15:11:22 2021 ] 	Top1: 21.37%

def parser(string):
    """

    :param string:
    :return: (key, val), key=0,1,2,3 (train epoch, eval epoch, train value, eval value)
    """
    token = ["Training epoch", "Eval epoch", "Mean training acc", "Top1"]
    for i, t in enumerate(token):
        try:
            index = string.index(t)
            if i == 0 or i == 1:
                s = string[index:]
                s = s.split(':')[1]
                return i, int(s.strip())
            if i == 2 or i == 3:
                s = string[index:]
                s = s.split(':')[1]
                return i, float(s.strip().replace('%', ''))
        except ValueError:
            continue
    return None


def verify(train_epoch, eval_epoch, train_acc, eval_acc):

    def clear_data(data):
        data.reverse()
        index = data.index(1)
        index = len(data) - index - 1
        data.reverse()
        return data[index:]

    train_epoch = clear_data(train_epoch)
    eval_epoch = clear_data(eval_epoch)
    train_acc = train_acc[len(train_acc) - len(train_epoch):]
    eval_acc = eval_acc[len(eval_acc) - len(eval_epoch):]
    return train_epoch, eval_epoch, train_acc, eval_acc


def load_log(filename):

    ret_list = [[], [], [], []]
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            ret = parser(line)
            if ret:
                ret_list[ret[0]].append(ret[1])
            line = f.readline()
    return verify(*ret_list)


def plot_acc_curve(data):
    color_list = ['#8dd3c7', '#bebada', '#fb8072', '#80b1d3', '#b3de69', '#fccde5', '#d9d9d9', '#bc80bd',
                  '#ccebc5', '#ffed6f', '#ffed6f']
    for i, item in enumerate(data):
        plt.plot(item["train_epoch"], item["train_acc"], color=color_list[i % len(color_list)],
                 marker='.', label=item["label"] + '-train')
        plt.plot(item["eval_epoch"], item["eval_acc"], color=color_list[i % len(color_list)],
                 marker='*', label=item["label"] + '-eval')
        plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    data = []
    train_epoch, eval_epoch, train_acc, eval_acc = load_log("../work_dir/fsd2/log.txt")
    data.append({
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "label": "ctrgcn"
    })
    train_epoch, eval_epoch, train_acc, eval_acc = load_log("../work_dir/fsd3/log.txt")
    data.append({
        "train_epoch": train_epoch,
        "eval_epoch": eval_epoch,
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "label": "ctrgcn2"
    })
    plot_acc_curve(data)

