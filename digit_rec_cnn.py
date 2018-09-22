import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mnist as mn
from PIL import Image, ImageFilter
import cv2

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape) :
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W) :
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

def get_normalized_data(data):
    data = data.as_matrix()
    data = data.astype(np.float32)
    tf.random_shuffle(data)
    x = data[:, 1:]
    mu = x.mean(axis = 0)
    std = x.std(axis = 0)
    np.place(std, std == 0,1)
    x = (x - mu) / std
    y = data[:,0]
    return x,y

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)

x_img = tf.reshape(x,[-1,28,28,1])

w_conv1 = weight_variable([5, 5, 1, 6])
b_conv1 = bias_variable([6])
h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

w_conv2 = weight_variable([5, 5, 6, 16])
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

w_fc1 = weight_variable([7*7*16, 120])
b_fc1 = bias_variable([120])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1)+b_fc1)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([120,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1,w_fc2)+b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)

correct_predection = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_predection,"float"))

init = tf.global_variables_initializer()

save_model = ".//model//mnist.ckpt"

def train(epoch) :
    train_images, train_labels, val_images, val_labels = mn.loadTrainData()
    n_samples = train_images.shape[0]
    batch_size = 50
    max_batch = (int)(n_samples / batch_size)
    with tf.Session() as sess:
        sess.run(init)
        start_time = time.time()
        c = []
        train_writer = tf.summary.FileWriter(".//log", sess.graph)  # 输出日志的地方
        saver = tf.train.Saver()
        for i in range(epoch):
            for batch in range(max_batch):
                batch_x = train_images[batch * batch_size: (batch + 1) * batch_size, :]
                batch_y = train_labels[batch * batch_size: (batch + 1) * batch_size, :]

                # batch_x = mn.train_images[batch * batch_size: (batch + 1) * batch_size, :]
                # batch_y = mn.train_labels[batch * batch_size: (batch + 1) * batch_size, :]
                sess.run(train_step, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})

            train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
            c.append(train_accuracy)
            loss = sess.run(cross_entropy, feed_dict={x: batch_x, y_: batch_y, keep_prob:1.0})
            print("traing accuracy %g" % (train_accuracy))
            print("cross entropy %f" % (loss))

            end_time = time.time()
            print("time: ", (end_time - start_time))
            start_time = end_time

            if i % 10 == 0 or i == epoch -1:
                saver.save(sess, save_model)
            print("---------------%d onpech is finished-------------------" % i)

            # print("validation accuracy %g" % accuracy.eval(feed_dict={
            #     x: mn.val_images, y_: mn.val_labels, keep_prob:1.0}))
            print("validation accuracy %g" % accuracy.eval(feed_dict={
                x: val_images, y_: val_labels, keep_prob: 1.0}))

        print("Model Save Finished!")
        plt.plot(c)
        plt.tight_layout()
        plt.savefig('cnn-tf-digit-rec2.png', dpi=200)
        plt.show()

def test():
    test_images = mn.loadTestData()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('.//model')
        saver.restore(sess, save_model)
        y_conv2 = sess.run(y_conv, feed_dict={x: test_images, keep_prob:1.0})
        y_test = (tf.argmax(y_conv2, 1)).eval(session=sess)

        submissions = pd.DataFrame({"ImageId": list(range(1, y_test.shape[0] + 1)),
                                    "Label": y_test})
        submissions.to_csv("DigitRecognizer3.csv", index=False, header=True)

def resize_img(file_name):
    image = cv2.imread(file_name)
    res = cv2.resize(image, (28, 28), interpolation=cv2.INTER_CUBIC)
    res  = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Image", res)
    # cv2.waitKey(0)
    res = np.reshape(res, [-1, 28*28])
    return res

def resize_img2(file_name):
    im = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    # 图片预处理
    # img_gray = cv2.cvtColor(im , cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 数据从0~255转为-0.5~0.5
    # img_gray = (im - (255 / 2.0)) / 255
    # cv2.imshow('out',img_gray)
    # cv2.waitKey(0)
    img_gray = (im - (255 / 2.0)) / 255
    x_img = np.reshape(img_gray, [-1, 784])
    return x_img

def recognize(file_name):
    saver = tf.train.Saver()
    result = resize_img2(file_name)
    with tf.Session() as sess:
        save_model = tf.train.latest_checkpoint('.//model')
        saver.restore(sess, save_model)
        prediction = tf.argmax(y_conv, 1)
        predict = prediction.eval(feed_dict={x: result, keep_prob: 1.0}, session=sess)

        print('recognize result:')
        print(predict[0])
        print(y_conv.eval(feed_dict={x: result, keep_prob: 1.0}, session=sess))

if __name__ == '__main__':
    # train(100)
    # test()
    recognize('input/test/test3.png')
    # resize_img2('input/test/test6.png')
