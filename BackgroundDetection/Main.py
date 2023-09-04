from GMM_Model import GMM
from ImageUtils import ImageUtils

'''
* 主函数Main
* author: holoxiong
* date: 2022.11.10
'''


if __name__ == '__main__':
    # 图片载入
    path = './WavingTrees'
    images = ImageUtils.load_image(path)

    # 划分训练集与测试集
    train_loader = images[:200]
    test_loader = images[200:]

    # 参数设置
    k = 5            # 混合高斯模型中高斯模型的个数
    lr = 0.01        # 模型学习率
    img = ImageUtils.read_image(images[0])
    sigma0 = 100     # 方差初始值
    weight0 = 0.005  # 权重初始值
    t = 0.9          # 阈值

    # 模型训练
    gmm = GMM(k, lr, img, sigma0, weight0, t)
    print('GMM training******')
    gmm.train(train_loader, show_img=False)
    # 模型测试
    print('GMM testing*******')
    gmm.test(test_loader, method='video', save_name='result_k'+str(gmm.k), show_img=False)
    gmm.test([images[247]], method='image', save_name='resultk' + str(gmm.k), show_img=False)
    print('GMM finished******')