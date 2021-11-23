import paddle
import random
import time
from loss import FocalLoss

start_time = time.time()
max_err = 0

for i in range(500):
    x = paddle.rand((1280, 2)) * random.randint(1, 10)
    x = paddle.create_parameter(shape=x.shape, dtype=str(x.numpy().dtype),
                                default_initializer=paddle.nn.initializer.Assign(x))
    label = paddle.randint(low=0, high=2, shape=[1280], dtype='int64')

    output0 = FocalLoss(alpha=None, gamma=0)(x, label)
    output1 = paddle.nn.CrossEntropyLoss()(x, label)
    a = output0.numpy()
    b = output1.numpy()
    if abs(a - b) > max_err:
        max_err = abs(a - b)

print('time:', time.time() - start_time, 'max_error:', max_err)
