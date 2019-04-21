# -*- coding: utf-8 -*-
"""
读取MNIST的csv数据，将其保存成图片，并转成tfrecords格式
"""

##################### load packages ######################
import tensorflow as tf
import numpy as np
import os
import cv2
##################### load data ##########################
from tensorflow.examples.tutorials.mnist import input_data


def convert_mnist_img(data, save_path):
    for i in range(data.images.shape[0]):
        img = data.images[i].reshape([28, 28, 1])
        img = (img * 255).astype(np.uint8)
        label = data.labels[i]
        # cv2.imshow('image', img)
        # cv2.waitKey(500)
        filename = save_path + '/{}_{}.jpg'.format(label, i)
        cv2.imwrite(filename, img)


def convert_img_tfrecords(data_path, record_dir):
    writer = tf.python_io.TFRecordWriter(record_dir)
    for file in os.listdir(data_path):
        img = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
        img_raw = img.tobytes()
        label = int(file.split('_')[0])
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data/')

    convert_mnist_img(mnist.train, 'img_train')
    print('convert training data to image complete')

    convert_mnist_img(mnist.test, 'img_test')
    print('convert test data to image complete')

    convert_img_tfrecords('./img_train', 'train_img.tfrecords')
    print('convert train image to tfrecords complete')

    convert_img_tfrecords('./img_test', 'test_img.tfrecords')
    print('convert test image to tfrecords complete')