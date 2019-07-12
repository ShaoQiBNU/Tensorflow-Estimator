# -*- coding: utf-8 -*-
"""
读取MNIST的tfrecords格式数据，然后利用cnn模型训练
高级estimator最终会将loss，accuracy 和 global_step等等 输出到 ./cnn_classifer_dataset/
可以调用tensorboard进行查看：tensorboard --logdir . --port 8889 --host 0.0.0.0
"""

##################### load packages ######################
import tensorflow as tf
import itertools

##################### 解析tfrecords ######################
def parse_data(example_proto):
    features = {'img_raw': tf.FixedLenFeature([], tf.string, ''),
                'label': tf.FixedLenFeature([], tf.int64, 0)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['img_raw'], tf.uint8)
    label = tf.cast(parsed_features['label'], tf.int64)
    image = tf.reshape(image, [28, 28, 1])
    image = tf.cast(image, tf.float32)
    return image, label

##################### 输入数据流 ######################
def my_input_fn(filenames, epochs, batch_size, mode=tf.estimator.ModeKeys.TRAIN):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(parse_data)

    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=50000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)

    else:
        dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

##################### 模型 ######################
def model_fn(features, labels, mode, params):
    '''
    :param features:  input image
    :param labels:    image label
    :param mode:      TRAIN or PREDICT
    :param params:    User-defined hyper-parameters, e.g. learning-rate.
    '''

    # First convolutional layer.
    net = tf.layers.conv2d(inputs=features, name='layer_conv1',
                           filters=16, kernel_size=5,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Second convolutional layer.
    net = tf.layers.conv2d(inputs=net, name='layer_conv2',
                           filters=36, kernel_size=5,
                           padding='same', activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=2, strides=2)

    # Flatten to a 2-rank tensor.
    net = tf.contrib.layers.flatten(net)

    # Eventually this should be replaced with:
    # net = tf.layers.flatten(net)

    # First fully-connected / dense layer.
    # This uses the ReLU activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc1',
                          units=128, activation=tf.nn.relu)

    # Second fully-connected / dense layer.
    # This is the last layer so it does not use an activation function.
    net = tf.layers.dense(inputs=net, name='layer_fc2',
                          units=10)

    # Logits output of the neural network.
    logits = net

    # Softmax output of the neural network.
    y_pred = tf.nn.softmax(logits=logits)

    # Classification output of the neural network.
    y_pred_cls = tf.argmax(y_pred, axis=1)

    # Define the loss-function to be optimized, by first
    # calculating the cross-entropy between the output of
    # the neural network and the true labels for the input data.
    # This gives the cross-entropy for each image in the batch.
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits)

    # Reduce the cross-entropy batch-tensor to a single number
    # which can be used in optimization of the neural network.
    loss = tf.reduce_mean(cross_entropy)


    # classification accuracy.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=y_pred_cls,
                                   name='acc_op')  # 计算精度

    metrics = {'accuracy': accuracy}  # 返回格式

    tf.summary.scalar('accuracy', accuracy[1])  # 仅为了后面图表统计使用


    if mode == tf.estimator.ModeKeys.EVAL:
        # If the estimator is supposed to be in prediction-mode
        # then use the predicted class-number that is output by
        # the neural network. Optimization etc. is not needed.

        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=metrics)


    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=y_pred_cls)
        return spec


    if mode == tf.estimator.ModeKeys.TRAIN:
        # Define the optimizer for improving the neural network.
        optimizer = tf.train.AdamOptimizer(learning_rate=params["learning_rate"])

        # Get the TensorFlow op for doing a single optimization step.
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Wrap all of this in an EstimatorSpec.
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op)

    return spec


if __name__ == '__main__':

    params = {"learning_rate": 1e-4}
    model = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir="./cnn_classifer_dataset/")

    # Train the Model
    input_fn = lambda: my_input_fn('train_img.tfrecords', 5, 256, tf.estimator.ModeKeys.TRAIN)
    train_results = model.train(input_fn, steps=2000)
    print(train_results)

    # Test the Model
    input_fn = lambda: my_input_fn('test_img.tfrecords', 1, 100, tf.estimator.ModeKeys.EVAL)
    predictions = model.evaluate(input_fn=input_fn)
    print('EVAL', predictions)
