import paddle
import paddle.nn as nn
from util.util import import_class
import paddle.nn.initializer as init
import numpy as np
import math


def conv_init(conv):
    if hasattr(conv, 'weight') and conv.weight is not None:
        init.KaimingNormal()(conv.weight)
        # nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if hasattr(conv, 'bias') and conv.bias is not None:
        init.Constant(0)(conv.bias)


def bn_init(bn, scale):
    init.Constant(scale)(bn.weight)
    init.Constant(0)(bn.bias)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            init.KaimingNormal()(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, paddle.Tensor):
            init.Constant(0)(m.bias)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init.Normal(mean=1.0, std=0.02)(m.weight)
            # m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            init.Constant(0)(m.bias)


class TemporalConv(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        # branches：different dilations, a simple conv and a max pool
        self.num_branches = len(dilations) + 2

        # 最后所有的branches concat 为out_channels
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)

        # Temporal Convolution branches
        self.branches = nn.LayerList([
            nn.Sequential(
                nn.Conv2D(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2D(branch_channels),
                nn.ReLU(),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2D(branch_channels),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2D(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2D(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2D(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        # 使用weights_init function 初始化每一个module
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = paddle.concat(branch_outs, axis=1)
        out += res
        return out


class CTRGC(nn.Layer):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        """

        :param in_channels:
        :param out_channels:
        :param rel_reduction: reduce scale of feature dimension
        :param mid_reduction:
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 3 or in_channels == 9:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction

        # full connection
        self.conv3 = nn.Conv2D(self.in_channels, self.out_channels, kernel_size=1)  # T
        self.conv1 = nn.Conv2D(self.out_channels, self.rel_channels, kernel_size=1)  # \psi
        self.conv2 = nn.Conv2D(self.out_channels, self.rel_channels, kernel_size=1)  # \phi
        self.conv4 = nn.Conv2D(self.rel_channels, self.out_channels, kernel_size=1)  # \xi
        self.tanh = nn.Tanh()

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                conv_init(layer)
            elif isinstance(layer, nn.BatchNorm2D):
                bn_init(layer, 1)

    def forward(self, x, A=None, alpha=1):
        """
        x: N,C,T,V
        A: V,V
        """

        # 对T个时间帧取均值
        # x1, x2: N,C,V
        # x3: N,C,T,V
        x3 = self.conv3(x)
        x1, x2 = self.conv1(x3).mean(-2), self.conv2(x3).mean(-2)

        # x1.unsqueeze(-1): N,C,V,1
        # $M_1(x_1, x_2) = \sigma(\psi(x_1) - \phi_(x_2))$
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))

        # $Q = \xi(M_1(x_1, x_2))$
        # $A + Q * \alpha$
        # x1: N*C'*V*V
        # A: V*V
        # A.unsqueeze(0).unsqueeze(0): 1*1*V*V
        # x1: N,C,V,V
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V

        # $x_3 = T(x)$: N,C',T,V
        # x_1: N,C',V,V
        # output: N,C',T,V
        # x1 = paddle.einsum('ncuv,nctv->nctu', x1, x3)
        x1 = paddle.matmul(x3, x1.transpose([0, 1, 3, 2]))
        return x1


class unit_tcn(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2D(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Layer):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        """

        :param in_channels:
        :param out_channels:
        :param A:
        :param coff_embedding:
        :param adaptive:
        :param residual:
        """
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]  # channels number
        self.convs = nn.LayerList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        # residual
        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2D(in_channels, out_channels, 1),
                    nn.BatchNorm2D(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        paddle_A = paddle.to_tensor(A.astype(np.float32))
        if self.adaptive:
            # learnable A
            self.PA = paddle.create_parameter(shape=paddle_A.shape, dtype=str(paddle_A.numpy().dtype),
                                              default_initializer=paddle.nn.initializer.Assign(paddle_A))
        else:
            self.A = paddle.create_parameter(shape=paddle_A.shape, dtype=str(paddle_A.numpy().dtype),
                                             default_initializer=paddle.nn.initializer.Assign(paddle_A))
            self.A.stop_gradient = True

        alpha = paddle.zeros([1])
        self.alpha = paddle.create_parameter(shape=alpha.shape, dtype=str(alpha.numpy().dtype),
                                             default_initializer=paddle.nn.initializer.Assign(alpha))
        self.bn = nn.BatchNorm2D(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU()

        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                conv_init(layer)
            elif isinstance(layer, nn.BatchNorm2D):
                bn_init(layer, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
            # A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z

        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class TCN_GCN_unit(nn.Layer):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations, residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Layer):
    def __init__(self, num_class=30, num_point=25, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super().__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # get adj of graph
        A = self.graph.A  # 3,25,25
        self.num_class = num_class
        self.num_point = num_point

        # batch norm for input data
        self.data_bn = nn.BatchNorm1D(num_person * in_channels * num_point)
        base_channel = 64

        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel*2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel*2, base_channel*2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel*2, base_channel*4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel*4, base_channel*4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel*4, num_class)
        init.Normal(0, math.sqrt(2. / num_class))(self.fc.weight)
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = x.reshape([N, T, self.num_point, -1]).transpose([0, 3, 1, 2]).unsqueeze(-1)
        N, C, T, V, M = x.shape

        x = x.transpose([0, 4, 3, 1, 2]).reshape([N, M * V * C, T])
        x.stop_gradient = False
        x = self.data_bn(x)
        x = x.reshape([N, M, V, C, T]).transpose([0, 1, 3, 4, 2]).reshape([N * M, C, T, V])
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.shape[1]
        x = x.reshape([N, M, c_new, -1])  # N, M, C', T*V
        x = x.mean(3).mean(1)  # N, C'
        x = self.drop_out(x)

        return self.fc(x)  # N, C' -> N, Class
