import os

import scipy.io as spio

dir_name = os.path.dirname(__file__)
data_file_name = os.path.join(dir_name, './mnist_data.mat')
label_file_name = os.path.join(dir_name, './mnist_label.mat')


def get_mnist_data_label():
    image_gallery = spio.loadmat(data_file_name)['data'].T
    true_label = spio.loadmat(label_file_name)['trueLabel'][0]
    return image_gallery, true_label
