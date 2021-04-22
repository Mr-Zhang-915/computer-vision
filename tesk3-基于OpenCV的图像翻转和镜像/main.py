import cv2
import numpy as np
import matplotlib.pyplot as plt


class ImageOpsFromScratch(object):
    def __init__(self, image_file):
        self.image_file = image_file    # 初始化图片路径
    # 定义读取图片函数，并转换成RGB矩阵
    def read_this(self, gray_scale=False):
        """
        默认情况下，imread()函数读取的图像BGR格式，要转换成常规RGB格式
        本函数从传入的图像文件返回图像矩阵
        :param gray_scale:
        :return:
        """
        image_src = cv2.imread(self.image_file) # 读取图片
        if gray_scale:
            image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY) # 转成灰度图
        else:
            image_rgb = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)  # 转成RGB矩阵
        return image_rgb
    # 定义画图函数
    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        """
        画图函数
        :param orig_matrix: 原RGB矩阵
        :param trans_matrix: 经过numpy反转后的矩阵
        :param head_text: 图像的标题
        :param gray_scale: 是否为灰度图
        :return:
        """
        fig = plt.figure()  # 定义一个画布
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.axis("off")
        ax1.title.set_text('Original')
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.axis('off')
        ax2.title.set_text(head_text)
        if not gray_scale:
            ax1.imshow(orig_matrix)
            ax2.imshow(trans_matrix)
            plt.show()
        else:
            ax1.imshow(orig_matrix, cmap='gray')
            ax2.imshow(trans_matrix, cmap='gray')
            plt.show()
        return True


    # 定义左右镜像函数
    def mirror_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_mirror = np.fliplr(image_rgb) # 利用numpy将矩阵左右翻转
        head_text = 'Mirrored'  # 图像标题
        if with_plot:
            self.plot_it(image_rgb, image_mirror, head_text=head_text, gray_scale=gray_scale)
            return None
        return image_mirror
    # 定义上下镜像函数
    def flip_this(self, with_plot=True, gray_scale=False):
        image_rgb = self.read_this(gray_scale=gray_scale)
        image_flip = np.flipud(image_rgb)  # 利用numpy将矩阵上下翻转
        head_text = 'Fliped'
        if with_plot:
            self.plot_it(image_rgb, image_flip, head_text=head_text, gray_scale=gray_scale)
            return None
        return image_flip

# 实例化类
imo = ImageOpsFromScratch(image_file='images/lena.jpg')
imo.mirror_this()   # 绘制左右镜像图
imo.mirror_this(gray_scale=True)    # 绘制左右镜像灰度图
imo.flip_this() # 绘制上下镜像图
imo.flip_this(gray_scale=True)  # 绘制上下镜像灰度图
