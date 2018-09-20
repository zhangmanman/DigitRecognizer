import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

def extract_images_and_labels(dataset, validation = False):
    #需要将数据转化为[image_num, x, y, depth]格式
    images = dataset[:, 1:]
    # images = dataset[:, 1:].reshape(-1, 28, 28, 1)

    #由于label为0~9,将其转化为一个向量.如将0 转换为 [1,0,0,0,0,0,0,0,0,0]
    labels_dense = dataset[:, 0]
    num_class = (int)(max(labels_dense) + 1)
    # num_labels = labels_dense.shape[0]
    labels_one_hot = tf.one_hot(labels_dense, num_class)
    labels_one_hot = tf.cast(labels_one_hot, tf.int32)
    sess = tf.InteractiveSession()
    labels_one_hot = labels_one_hot.eval(session=sess)
    sess.close()
    if validation:
        num_images = images.shape[0]
        divider = num_images - 200
        return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]
    else:
        return images, labels_one_hot

def extract_images(dataset):
    return dataset.reshape(-1, 28*28)


def loadData():
    train_data_file = 'input/all/train.csv'
    test_data_file = 'input/all/test.csv'

    train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
    test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)

    tf.random_shuffle(train_data)

    train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
    test_images = extract_images(test_data)


train_data_file = 'input/all/train.csv'
test_data_file = 'input/all/test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.uint8)
test_data = pd.read_csv(test_data_file).as_matrix().astype(np.uint8)

tf.random_shuffle(train_data)

train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data, validation=True)
test_images = extract_images(test_data)