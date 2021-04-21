import cv2
import matplotlib.pyplot as plt
import numpy as np

# 1、读取测试图片
image = cv2.imread('./image/barcodes.jpg', cv2.IMREAD_GRAYSCALE)    # 读取一副灰度图
image_out = cv2.imread('image/barcodes.jpg')


# 2、对图像进行二值化处理
# print(image.shape)    # (426, 600)
scale = 800.0 / image.shape[1]
image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
# print(image.shape)  # (568, 800)

# 使用黑帽运算符+阈值处理
kernel = np.ones((1,3), np.uint8)   # 创建一个一行三列的1矩阵  [[1,1,1]]
image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel, anchor=(1,0))   # 黑帽运算
thresh, image = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)    # 阈值处理，此处设置阈值为10，填充色为255，阈值类型设为二值化化，大于阈值使用255填充，小于阈值设为0


# 膨胀+闭运算
kernel = np.ones((1,5), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2)  # 膨胀运算
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)   # 闭运算

# 使用35*21的内核打开
kernel = np.ones((21,35), np.uint8)
image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
# plt.imshow(image, cmap='Greys_r')
# plt.show()
# 检测二维码的轮廓 contours:轮廓本身，type：list; hierarchy:每条轮廓对应的属性，type:ndarray
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 只检测外轮廓，存储所有的轮廓点，相邻的两个点的像素差不超过1

# 过滤噪声
unscale = 1.0 / scale
if contours != None:    # 判断检测出的轮廓列表是否为空
    for contour in contours:    # 循环读取每个轮廓
        if cv2.contourArea(contour) <= 2000:    # 计算轮廓面积，如果小于2000，就丢弃掉，继续读取下一个轮廓
            continue
        rect = cv2.minAreaRect(contour) # 计算每个轮廓包含点集的最小面积
        # minAreaRect()返回一个Box2D结构rect：（最小外接矩形的中心（x,y），（宽度，高度），旋转角度）
        rect = ((int(rect[0][0] * unscale), int(rect[0][1] * unscale)),
                (int(rect[1][0] * unscale), int(rect[1][1] * unscale)),
                rect[2])
        box = np.int0(cv2.boxPoints(rect))  # 获取到外接矩形的四个顶点坐标
        cv2.drawContours(image_out, [box], 0, (0, 255, 0), thickness=2) # 绘制轮廓，（指明在哪幅图片上绘制轮廓，轮廓本身-是一个list，指明绘制轮廓的list中的哪条轮廓，线条颜色，线条宽度）


plt.imshow(image_out)
cv2.imwrite('image/out.jpg', image_out)