import paddle


def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = paddle.zeros([rot.shape[0], 1])  # T,1
    ones = paddle.ones([rot.shape[0], 1])  # T,1

    r1 = paddle.stack((ones, zeros, zeros), axis=-1)  # T,1,3
    rx2 = paddle.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), axis=-1)  # T,1,3
    rx3 = paddle.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), axis=-1)  # T,1,3
    rx = paddle.concat((r1, rx2, rx3), axis=1)  # T,3,3

    ry1 = paddle.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), axis=-1)
    r2 = paddle.stack((zeros, ones, zeros), axis=-1)
    ry3 = paddle.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), axis=-1)
    ry = paddle.concat((ry1, r2, ry3), axis=1)

    rz1 = paddle.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), axis=-1)
    r3 = paddle.stack((zeros, zeros, ones), axis=-1)
    rz2 = paddle.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), axis=-1)
    rz = paddle.concat((rz1, rz2, r3), axis=1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_batch, theta=0.3):
    """
    data_paddle: C,T,V,M
    """
    B, C, T, V, M = data_batch.shape
    data_batch_new = None
    for data_paddle in data_batch:
        data_paddle = data_paddle.transpose([1, 0, 2, 3]).reshape([T, C, V * M])  # T,3,V*M
        rot = paddle.uniform([3], min=-theta, max=theta)
        rot = paddle.stack([rot, ] * T, axis=0)
        rot = _rot(rot)  # T,3,3
        data_paddle = paddle.matmul(rot, data_paddle)
        data_paddle = data_paddle.reshape([T, C, V, M]).transpose([1, 0, 2, 3])
        data_paddle = data_paddle.unsqueeze(0)
        data_batch_new = paddle.concat([data_batch_new, data_paddle], axis=0) if data_batch_new is not None else data_paddle

    return data_batch_new
