import os

from datetime import datetime
# 导入opencv-python
import cv2


def gettime():
    return datetime.now().strftime("%Y_%m_%d %H:%M:%S.%f")


class Biophoto:
    def __init__(self, dir):
        # 判断文件夹时候存在
        if os.path.isdir(f"{dir}\\bioface"):
            pass
        else:
            # 文件不存在则创建
            os.makedirs(f"{dir}\\bioface")
        # 获取目录下所有图片返回列表
        self.listdir = os.listdir(dir)
        # 创建日志记录文件
        self.logs = open(f"{dir}\\Bioface{gettime()[:10]}.txt", "a", encoding='utf8')

    def getface(self, dir):
        # 导入人脸级联分类器引擎，'.xml'文件里包含训练出来的人脸特征
        face_engine = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # 过滤非jpg文件
        photonames = filter(lambda x: x.endswith('jpg'), self.listdir)
        for photoname in photonames:
            img = cv2.imread(f'{dir}\\' + photoname, 1)
            # cv2.IMREAD_COLOR：默认参数，读入一副彩色图片，忽略alpha通道，可用1作为实参替代
            # cv2.IMREAD_GRAYSCALE：读入灰度图片，可用0作为实参替代
            # cv2.IMREAD_UNCHANGED：顾名思义，读入完整图片，包括alpha通道，可用 -1
            try:
                # 用人脸级联分类器引擎进行人脸识别，返回的faces为人脸坐标列表，1.3是放大比例，5是重复识别次数
                faces = face_engine.detectMultiScale(img, scaleFactor=1.4, minNeighbors=5)
                print(f"{photoname}人脸坐标", faces)
            except Exception as e:
                self.logs.write(f"{gettime()} {photoname} error:{e}。\n")
                print(e)

            # 对每一张脸，进行如下操作
            try:
                if len(faces) > 0:
                    for (x, y, w, h) in faces:  # 人脸坐标
                        cropped_image = img[y:y + w, x:x + h]  # 裁剪坐标
                        cv2.imwrite(f"{dir}\\bioface\\verify_biophoto_9_" + photoname, cropped_image)  # 保存裁剪
                        self.logs.write(f"{gettime()} {photoname} 读取人脸成功。\n")
                else:
                    self.logs.write(f"{gettime()} {photoname} 读取人脸失败----------\n")
            except Exception as e:
                print(f"人脸裁剪失败{e}")
        self.logs.close()


if __name__ == "__main__":
    # dir=input("输入照片路径：") # 最好用绝对路径
    dir = "photos"
    s = Biophoto(dir)
    s.getface(dir)
