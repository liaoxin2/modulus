import paddle
import numpy as np


def reshape_fields(img, inp_or_tar, y_roll, train, n_history, in_channels,
    out_channels, img_shape_x, img_shape_y, min_path, max_path,
    global_means_path, global_stds_path, normalization, roll, normalize=True):
    """
    Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of
    size ((n_channels*(n_history+1), img_shape_x, img_shape_y)
    """
    if len(np.shape(img)) == 3:
        img = np.expand_dims(img, 0)
    if tuple(img.shape)[3] > 720:
        img = img[:, :, 0:720]
    n_history = n_history
    n_channels = np.shape(img)[1]
    channels = in_channels if inp_or_tar == 'inp' else out_channels
    if normalize and train:
        mins = np.load(min_path)[:, channels]
        maxs = np.load(max_path)[:, channels]
        means = np.load(global_means_path)[:, channels]
        stds = np.load(global_stds_path)[:, channels]
    img = img[:, :, :img_shape_x, :img_shape_y]
    if normalize and train:
        if normalization == 'minmax':
            img -= mins
            img /= maxs - mins
        elif normalization == 'zscore':
            img -= means
            img /= stds
    if roll:
        img = np.roll(img, y_roll, axis=-1)
    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels * (n_history + 1), img_shape_x,
            img_shape_y))
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (n_channels, img_shape_x, img_shape_y))
    return paddle.to_tensor(data=img)
