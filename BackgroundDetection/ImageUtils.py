import os
import cv2
import numpy as np

'''
* ImageUtils:图片的读取与展示
* author: holoxiong
* date: 2022.11.10
'''


class ImageUtils:
    # 载入图片
    @staticmethod
    def load_image(path):
        images = []
        # 读取路径下的所有
        for image_name in os.listdir(path):
            if image_name.endswith('bmp'):
                images.append(path + '/' + image_name)
        images = np.sort(images).tolist()
        return images

    # 读取图片
    @staticmethod
    def read_image(image):
        image = cv2.imread(image)
        image = image.astype(float)
        return image

    # 图片显示
    @staticmethod
    def show_images(*images, x=1, y=1, size=(160, 120), scale=1, name='Figure', show_img=False):
        img_index = 0
        # 图像尺寸取整
        scaled_size = (np.ceil(size[0] * scale).astype(np.int), np.ceil(size[1] * scale).astype(np.int))

        # 利用cv2绘制一个面板 展示背景、前景检测的结果
        for i in range(x):
            for j in range(y):
                if img_index >= len(images):
                    add_img = cv2.resize((images[0] * 0 + 255).astype(np.uint8), scaled_size)
                else:
                    add_img = cv2.resize(images[img_index].astype(np.uint8), scaled_size)
                if j == 0:
                    ylabel_imgs = add_img
                else:
                    ylabel_imgs = np.hstack([ylabel_imgs, add_img])
                img_index += 1
            if i == 0:
                imgbox = ylabel_imgs
            else:
                imgbox = np.vstack([imgbox, ylabel_imgs])

        # 展示一张一张图片的测试过程，需要键盘按任意键继续
        if show_img:
            cv2.namedWindow(name)
            cv2.imshow(name, imgbox)
            cv2.moveWindow(name, 200, 50)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        return imgbox

#
# if __name__ == '__main__':
#     images = ImageUtils.load_image('./WavingTrees')
#     print(images)
