import os

import numpy as np
from PIL import Image
import tensorflow as tf


def load_image(path, size=None):
  """sizeがNoneのときは画像のそのままのサイズで読み込む"""
  img = Image.open(os.path.expanduser(path)).convert("RGB")
  if size is not None:
    w = img.width
    h = img.height
    l = max(size[0], size[1])
    if w < h:
      rate = 0. + h / w
      w = l
      h = l * rate
    else:
      rate = 0. + w / h
      w = l * rate
      h = l
    size[0] = max(int(w), 1)
    size[1] = max(int(h), 1)
    img = img.resize(size, Image.LANCZOS)
  return tf.constant(transform_for_train(np.array([np.array(img)[:, :, :3]], dtype=np.float32)))


def transform_for_train(img):
  """
  読み込む画像がRGBなのに対し、VGGなどのパラメータがBGRの順なので、順番を入れ替える。
  ImageNetの色の平均値を引く。
  """
  return img[..., ::-1] - 120


def transform_from_train(img):
  """
  transform_for_trainの逆操作。
  """
  data = img[:, :, ::-1] + 120
  return data.clip(0, 255)
