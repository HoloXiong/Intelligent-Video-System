import os
import numpy as np
import cv2
from tqdm import tqdm


class ImageUtils:
    """
    func:图片工具类
    author:holoxiong
    date:2022.12.3
    """
    # 载入图片
    @staticmethod
    def load_image(path):
        images = []
        # 读取路径下的所有
        for image_name in os.listdir(path):
            if image_name.endswith('jpg'):
                images.append(path + '/' + image_name)
        images = np.sort(images).tolist()
        return images

    # 读取图片
    @staticmethod
    def read_image(image):
        image = cv2.imread(image)
        return image

    # 图片转视频
    @staticmethod
    def images2video(images):
        fps = 15
        size = (768, 576)   # 注意是（宽，高）
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter()
        video.open('./result.mp4', fourcc, fps, size)
        for image in tqdm(images):
            video.write(image)
        video.release()


if __name__ == '__main__':
    filenames = ImageUtils.load_image('./Walking/img')
    print(filenames)
    images = []
    for file in filenames:
        image = ImageUtils.read_image(file)
        images.append(image)
    ImageUtils.images2video(images)