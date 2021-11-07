import tensorflow as tf
import numpy as np
from chainer.links import caffe
from tensorflow.keras.mixed_precision import experimental as mixed_precision


class Conv:
  def __init__(self, chainer_conv):
    W = chainer_conv.W.data
    b = chainer_conv.b.data
    self.W = tf.constant(np.transpose(W, [2, 3, 1, 0]))
    self.b = tf.constant(b)

  def __call__(self, x, stride=1, activation_fn=tf.nn.relu, padding="SAME"):
    policy = mixed_precision.Policy('mixed_float16')
    y = tf.nn.conv2d(x, self.W, strides=[1, stride, stride, 1], padding=padding) + self.b
    return activation_fn(y) if activation_fn else y


def load_caffemodel(caffemodel):
  print("load model... %s" % caffemodel)
  model = caffe.CaffeFunction(caffemodel)
  return lambda layer_name: Conv(getattr(model, layer_name))
