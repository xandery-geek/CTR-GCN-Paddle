import paddle
from paddle import nn
import numpy as np


class FocalLoss(nn.Layer):
    def __init__(self, alpha, gamma=2.0, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = paddle.to_tensor([alpha, 1 - alpha])
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = paddle.to_tensor(alpha)
        self.size_average = size_average

    @staticmethod
    def _gather(x, index, dim=0):
        out = None
        if dim == 0:
            for i in range(len(index)):
                out = paddle.concat((out, x[index[i]].unsqueeze(0)), axis=0) \
                    if out is not None else x[index[i]].unsqueeze(0)
        if dim == 1:
            for i in range(len(index)):
                out = paddle.concat((out, x[i][index[i]].unsqueeze(0)), axis=0) \
                    if out is not None else x[i][index[i]].unsqueeze(0)
        return out

    def forward(self, pred, label):
        assert pred.dim() == 2
        if label.dim() == 1:
            label = label.reshape((-1, 1))
        assert label.dim() == 2

        log_pt = nn.functional.log_softmax(pred)  # calculate probability using softmax
        log_pt = self._gather(log_pt, label, dim=1)  # gather value of positive samples
        log_pt = log_pt.reshape(paddle.to_tensor([-1], dtype='int32'))  # B, C -> B*C

        log_pt_exp = log_pt.exp()
        pt = paddle.create_parameter(shape=log_pt_exp.shape, dtype=str(log_pt_exp.numpy().dtype),
                                     default_initializer=paddle.nn.initializer.Assign(log_pt_exp))

        if self.alpha is not None:
            if self.alpha.dtype != pred.dtype:
                self.alpha = self.alpha.astype(pred.dtype)
            at = self._gather(self.alpha, label.reshape(paddle.to_tensor([-1], dtype='int32')), dim=0)
            at = paddle.create_parameter(shape=at.shape, dtype=str(at.numpy().dtype),
                                         default_initializer=paddle.nn.initializer.Assign(at))
            log_pt = log_pt * at

        loss = -1 * (1 - pt) ** self.gamma * log_pt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
