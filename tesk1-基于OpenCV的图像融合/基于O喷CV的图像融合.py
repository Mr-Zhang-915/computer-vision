import cv2

# 1、读取图片
bg = cv2.imread('images/background.jpg', cv2.IMREAD_COLOR)  # 读取背景图片
fg = cv2.imread('images/foreground.jpg', cv2.IMREAD_COLOR)  # 读取内容图片

# 2、图像预处理
# 调整图片大小，图像融合需要两张图片大小相同
# print(bg.shape)
# print(fg.shape)
dim = (1200, 800)   # 定义图片大小为800到1200像素
resized_bg = cv2.resize(bg, dim, interpolation=cv2.INTER_AREA)
resized_fg = cv2.resize(fg, dim, interpolation=cv2.INTER_AREA)

# 3、混合图像
# 背景图片权重为0.5， 前景图片权重为0.8，是的背景更暗，前景更亮
blend = cv2.addWeighted(resized_bg, 0.5, resized_fg, 0.8, 0.0)

# 4、导出图片
cv2.imwrite('blended.jpg', blend)

