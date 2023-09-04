import cv2
import numpy as np
from tqdm import tqdm
from ImageUtils import ImageUtils

'''
* 利用GMM Model实现一个背景模型，用于检测Waving Trees的前景对象
* author: holoxiong
* date: 2022.11.10
'''


class GMM:
    def __init__(self, k, lr, img, sigma0, weight0, t):
        self.k = k  # 混合高斯模型中高斯模型的个数
        self.lr = lr  # 模型学习率
        self.img_size = img.shape  # 图片的尺寸
        self.sigma0 = sigma0  # 方差初始值
        self.weight0 = weight0  # 权重初始值
        self.t = t  # 阈值
        self.mean_matrix, self.sigma, self.weight = self.init_GMM(img)  # 均值均值，方差矩阵，权重矩阵

    # 初始化 GMM 模型
    def init_GMM(self, img):
        zero_matrix = np.zeros(self.img_size, dtype=float)  # 创建一个与img大小相同的0矩阵 eg:(1, 2, 3)
        zero_matrix = np.expand_dims(zero_matrix, axis=0)  # 扩展成三维矩阵 (1, 1, 2, 3)
        zero_matrix = np.repeat(zero_matrix, self.k - 1, axis=0)  # 矩阵重复 k=7 => (6, 1, 2, 3)

        one_matrix = np.ones(self.img_size, dtype=float)  # 创建一个与img大小相同的1矩阵
        one_matrix = np.expand_dims(one_matrix, axis=0)  # 扩展成三维矩阵

        # 均值矩阵
        mean_matrix = np.expand_dims(img, axis=0)
        mean_matrix = np.concatenate((mean_matrix, zero_matrix), axis=0)
        # sigma方差矩阵
        sigma = np.concatenate((self.sigma0 * one_matrix, zero_matrix), axis=0)
        # weight权重矩阵
        weight = np.concatenate((one_matrix, zero_matrix), axis=0)
        return mean_matrix, sigma, weight

    # 匹配高斯分布
    def match_gaussian(self, img, first_match=True, mask=None):
        delt = np.square(self.mean_matrix - img) < (self.sigma * 6.25)  # boolean矩阵
        match = np.sum(delt, axis=-1).astype(float)
        match[match < 2.5] = 0.0
        match[match > 2.5] = 1.0
        match = np.expand_dims(match, axis=-1)
        match = np.repeat(match, 3, axis=-1)

        # 只保留第一个匹配的位置
        if first_match:
            temp = np.zeros(match[0].shape)
            for i in range(0, match.shape[0]):
                add = temp + match[i]
                match[i] -= temp
                temp = add
            match[match > 0.1] = 1.0
            match[match <= 0.1] = 0.0

        # 由阈值遮罩矩阵得到unmatched
        if mask is not None:
            unmatched = 1.0 - np.max(match * mask, axis=0)
        else:
            unmatched = 1.0 - np.max(match, axis=0)
        return match, unmatched

    # 更新GMM model
    def update_model(self, img, match, unmatched):
        # 计算第k个分模型（根据分模型公式）
        left = 1 / (np.sqrt(2 * 3.14 * self.sigma + 1e-10))
        right = np.exp(-0.5 * np.square(self.mean_matrix - img) / (self.sigma + 1e-10))
        fai = self.lr * (left * right)

        # 更新均值矩阵
        self.mean_matrix = (1.0 - match) * self.mean_matrix + \
                           match * ((1.0 - fai) * self.mean_matrix + fai * img)
        # 更新方差矩阵
        self.sigma = (1.0 - match) * self.sigma + \
                     match * ((1.0 - fai) * self.sigma +
                              fai * np.square(self.mean_matrix - img))
        # 更新权重矩阵
        self.weight = (1 - self.lr) * self.weight + \
                      self.lr * match

        # 从小到大排序
        mean_sigma = np.mean(self.sigma)
        mean_sigma = np.expand_dims(mean_sigma, axis=-1)
        min_index = np.argsort(-(self.weight / (mean_sigma + 1e-10)), axis=0)
        self.weight = np.take_along_axis(self.weight, min_index, axis=0)
        self.mean_matrix = np.take_along_axis(self.mean_matrix, min_index, axis=0)
        self.sigma = np.take_along_axis(self.sigma, min_index, axis=0)

        self.mean_matrix[-1] = img * unmatched + (1 - unmatched) * self.mean_matrix[-1]
        one_matrix = np.ones(self.img_size, dtype=float)
        self.sigma[-1] = np.expand_dims(one_matrix, axis=0) * self.sigma0 * unmatched + \
                         (1 - unmatched) * self.sigma[-1]
        self.weight[-1] = np.expand_dims(one_matrix, axis=0) * self.weight0 * unmatched + \
                          (1 - unmatched) * self.weight[-1]
        self.weight = self.weight / (np.sum(self.weight, axis=0) + 1e-10)

    # 生成阈值遮罩矩阵
    def generate_mask(self):
        mask = np.zeros(self.weight.shape)
        mask[0] = self.weight[0]
        for i in range(1, mask.shape[0]):
            mask[i] = mask[i - 1] + self.weight[i]
        for i in range(mask.shape[0] - 2, -1, -1):
            mask[i + 1] = mask[i]
        mask[mask > self.t] = 1
        mask[mask <= self.t] = 0
        mask = 1 - mask
        mask[0] = np.ones(mask[0].shape)
        return mask

    # 模型训练
    def train(self, images, show_img):
        for img in tqdm(images):
            image = ImageUtils.read_image(img)
            # 匹配
            match, unmatched = self.match_gaussian(image, True)
            # 更新
            self.update_model(image, match, unmatched)
            # 显示
            ImageUtils.show_images(image, unmatched * 255, 2, 1, (image.shape[1], image.shape[0]), 2, show_img)

    # 模型测试
    def test(self, images, method, save_name, show_img):
        # 输出结果视频
        if method == 'video':
            fps = 8  # 帧率
            size = (self.img_size[1] * 4, self.img_size[0] * 4)
            fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
            output = cv2.VideoWriter()
            output.open('./'+save_name+'.mp4', fourcc, fps, size)

        # 生成遮罩矩阵
        mask = self.generate_mask()

        kernel_ero = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # MORPH_ELLIPSE
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # MORPH_RECT

        # 测试过程
        for img in tqdm(images):
            image = ImageUtils.read_image(img)
            match, unmatched = self.match_gaussian(image, first_match=False, mask=mask)
            ero = cv2.erode(unmatched, kernel_ero)
            dil = cv2.dilate(ero, kernel_dil)
            padding = dil * image
            image_box = ImageUtils.show_images(image, unmatched * 255, dil * 255, padding, x=2, y=2,
                                               size=(image.shape[1], image.shape[0]), scale=2, show_img=show_img)

            # 按方式输出结果
            if method == 'video':
                output.write(image_box)
            if method == 'image':
                res_pos = './'+save_name+'.png'
                cv2.imwrite(res_pos, image_box)

        if method == 'video':
            output.release()


if __name__ == '__main__':
    zero_matrix = np.zeros((60, 80, 3), dtype=float)  # 创建一个与img大小相同的0矩阵 eg:(2, 3)
    print(zero_matrix.shape)
    zero_matrix = np.expand_dims(zero_matrix, axis=0)  # 扩展成三维矩阵 (1, 2, 3)
    print(zero_matrix.shape)
    zero_matrix = np.repeat(zero_matrix, 7 - 1, axis=0)  # 矩阵重复 k=7 => (6, 2, 3)
    print(zero_matrix.shape)