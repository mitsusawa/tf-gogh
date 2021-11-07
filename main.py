import os

import models
from data import load_image
from config import Config, parse_args


def main():
  args = parse_args()
  config = Config(args)

  # 出力先の作成
  os.makedirs(config.output_dir, exist_ok=True)



  # 画像サイズの修正
  img_orig = load_image(config.original_image, [config.width, config.height])
  img_style = load_image(config.style_image, [config.width, config.height] if not config.no_resize_style else None)

  config.width = img_orig[0].shape[0]
  config.height = img_orig[0].shape[1]
  config.output_shape = [config.batch_size, config.width, config.height, 3]

  # print(config.width)
  # print(config.height)

  # モデルの作成
  model = models.generate_model(config.model)

  # 画像を生成
  generator = models.Generator(model, img_orig, img_style, config)
  generator.generate(config)


if __name__ == '__main__':
  main()
