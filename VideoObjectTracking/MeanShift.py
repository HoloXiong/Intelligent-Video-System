import numpy as np
import cv2
from tqdm import tqdm
from ImageUtils import ImageUtils


class MeanShift:
    """
    func:基于均值漂移的目标跟踪
    author:holoxiong
    date:2022.12.3
    """

    def __init__(self, x, y, w, h):
        self.x = x  # 跟踪目标在第一帧的位置
        self.y = y
        self.w = w
        self.h = h
        self.frames = self.init_msf()

    # 初始化视频序列
    def init_msf(self):
        filenames = ImageUtils.load_image('./Walking/img')
        images = []
        for file in filenames:
            image = ImageUtils.read_image(file)
            images.append(image)
        return images

    def get_target(self):
        # 取第一帧图片
        first_frame = self.frames[0]
        # 圈出第一个目标位置
        x = self.x
        y = self.y
        w = self.w
        h = self.h

        # 用于显示目标位置
        img = cv2.rectangle(first_frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        # cv2.imshow('First Frame', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)

        # 切出目标对象所在的区域
        target = first_frame[y:y + h, x:x + w]
        return img, target

    def get_weight_matrix(self, a, b):
        """
        计算权值矩阵
        :param a: 目标shape(0)
        :param b: 目标shape(1)
        :return: 权值矩阵weight_matrix
        """
        weight_matrix = np.zeros((a, b))
        # 每个位置相对于中心点(a/2, b/2)的权值计算
        for i in range(a):
            for j in range(b):
                # Epannechnikov核函数
                weight = (i - a / 2) ** 2 + (j - b / 2) ** 2
                weight_matrix[i, j] = 1 - weight / ((a / 2) ** 2 + (b / 2) ** 2)
        return weight_matrix

    def get_weight_hist(self, a, b, target, weight_matrix, coeff):
        """
        计算权值直方图
        :param weight_matrix: 权值矩阵
        :param a: 目标shape(0)
        :param b: 目标shape(1)
        :param target: 目标图像
        :param coeff: 归一化系数
        :return: 权值直方图weight_hist
        """
        weight_hist = np.zeros(4096)
        temp_layer = np.zeros((a, b))
        for i in range(a):
            for j in range(b):
                # 将rgb颜色空间量化为16*16*16
                r_layer = np.floor(float(target[i, j, 0]) / 16)
                g_layer = np.floor(float(target[i, j, 1]) / 16)
                b_layer = np.floor(float(target[i, j, 2]) / 16)
                temp_layer[i, j] = r_layer * 256 + g_layer * 16 + b_layer
                weight_hist[int(temp_layer[i, j])] += weight_matrix[i, j]
        # 乘以归一化系数
        weight_hist *= coeff
        return weight_hist, temp_layer

    # 主函数
    def main(self):
        # 获得目标
        img, target = self.get_target()
        # 带有矩形框的视频帧
        new_frames = []
        new_frames.append(img)
        a, b, c = target.shape
        # 计算目标的权值矩阵
        weight_matrix = self.get_weight_matrix(a, b)
        # 计算目标的权值直方图
        coeff = 1 / sum(sum(weight_matrix))
        weight_hist, _ = self.get_weight_hist(a, b, target, weight_matrix, coeff)

        # 开始目标跟踪mean shift迭代
        for i in range(1, len(self.frames)):
            epoch = 0
            print('正在跟踪第{}帧'.format(i))
            # 跟踪目标1
            while (a / 2) ** 2 + (b / 2) ** 2 > 0.5 and epoch < 20:
                epoch += 1
                new_target = self.frames[i][self.y:self.y + self.h, self.x:self.x + self.w]
                # 候选区域直方图
                new_weight_hist, temp_layer = self.get_weight_hist(a, b, new_target, weight_matrix, coeff)

                # 计算各个点的权重值
                weight = np.zeros(4096)
                for j in range(4096):
                    if new_weight_hist[j] != 0:
                        weight[j] = np.sqrt(weight_hist[j] / new_weight_hist[j])
                    else:
                        weight[j] = 0

                # 计算漂移的距离
                total_weight = 0
                total_xweight = 0
                for m in range(a):
                    for n in range(b):
                        total_weight += weight[int(temp_layer[m, n])]
                        total_xweight += weight[int(temp_layer[m, n])] * np.array([m - a / 2, n - b / 2])
                shift = total_xweight / total_weight

                # 更新矩形框的位置
                self.x += shift[1]
                self.x = int(self.x)
                self.y += shift[0]
                self.y = int(self.y)
            # 画矩形
            img = cv2.rectangle(self.frames[i], (self.x, self.y), (self.x + self.w, self.y + self.h), (255, 0, 0), 2)
            new_frames.append(img)
            # cv2.imshow('First Frame', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
        # 生成视频
        ImageUtils.images2video(new_frames)
        print('Finished')


if __name__ == '__main__':
    msf = MeanShift(686, 436, 33, 81)
    msf.main()
