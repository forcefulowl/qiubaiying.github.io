---
layout: post
title:  图像处理
subtitle:   Image Processing
date:   2018-02-15
author: gavin
header-img: img/post-bg-coffee.jpeg
catalog:    true
tags:
    - python
---

>大年三十来一发！

# 图像处理部分

### RGB --> Gray-Scale

```python
def rgbToGray(img):
    row, column, channel = img.shape
    for i in range(row):
        for j in range(column):
            # print('row = ',i,'column = ',j)
            avglist = img[i,j]
            avg = int(avglist[0]*0.299 + avglist[1]*0.587 + avglist[2]*0.114)
            img[i,j] = [avg,avg,avg]
    return img
```
### Histogram

```python
def histogram(img):
    row,column,channel = img.shape

    list1 = []
    list2 = []

    for i in range(256):
        list1.append(0)

    for i in range(256):
        list2.append(i)

    for i in range(row):
        for j in range(column):
            list1.append(img[i][j][0])

    list1 = list1[:-1]
    plt.hist(list1, bins=list2)
    plt.show()
```

### Average Smooth

```python
def smooth_avg(img):
    row, column, channel = img.shape
    img1 = deepcopy(img)
    for i in range(1,row-1):
        for j in range(1,column-1):
            img[i,j,0] = math.floor(img1[i,j,0]*0.2)+math.floor(img1[i-1,j,0]*0.2)+math.floor(img1[i,j-1,0]*0.2)+math.floor(img1[i,j+1,0]*0.2)+math.floor(img1[i+1,j,0]*0.2)
            #img[i,j,1] = math.floor(img1[i,j,1]*0.2)+math.floor(img1[i-1,j,1]*0.2)+math.floor(img1[i,j-1,1]*0.2)+math.floor(img1[i,j+1,1]*0.2)+math.floor(img1[i+1,j,1]*0.2)
            #img[i,j,2] = math.floor(img1[i,j,2]*0.2)+math.floor(img1[i-1,j,2]*0.2)+math.floor(img1[i,j-1,2]*0.2)+math.floor(img1[i,j+1,2]*0.2)+math.floor(img1[i+1,j,2]*0.2)
            img[i,j,1] = img[i,j,2] = img[i,j,0]
    return img

```

### Salt-Pepper

```python
def salt_pepper(img):
    row, column, channel = img.shape
    nums = int((row * column)*0.1)
    for i in range(nums):
        num1 = random.randint(0,row-1)
        num2 = random.randint(0,column-1)
        if img[num1,num2,0]>=100:
            img[num1,num2] = [0,0,0]
        else:
            img[num1,num2] = [255,255,255]
    return img
```
### Median Filtering

```python
def smooth_median(img):
    row, column, channel = img.shape
    img1 = deepcopy(img)
    for i in range(1,row-1):
        for j in range(1,column-1):
            list = [img1[i-1,j-1,0],img1[i-1,j,0],img1[i-1,j+1,0],img1[i,j-1,0],img1[i,j,0],img1[i,j+1,0],img1[i+1,j-1,0],img1[i+1,j,0],img1[i+1,j+1,0]]
            list.sort()
            img[i,j,0] = img[i,j,1] = img[i,j,2] = list[4]
    return img
```
### Binary Thresholding

```python 
def binary_thresholding(img, thresholding):
    row, column, channel = img.shape
    for i in range(row):
        for j in range(column):
            if img[i,j,0] >= thresholding:
                img[i,j] = [255,255,255]
            else:
                img[i,j] = [0,0,0]
    return img
```
### Binary P-Tile

```python
def binary_ptile(img,x):
    row, column, channel = img.shape
    nums = row * column
    list = []
    for i in range(row):
        for j in range(column):
            list.append(img[i,j,0])
    list.sort()
    p = list[int((nums-1)*(1-x*0.01))]
    for i in range(row):
        for j in range(column):
            if img[i,j,0] >= p:
                img[i,j] = [255,255,255]
            else:
                img[i,j] = [0,0,0]
    return img
```
### Iterative Thresholding

```python
def calculateT(img,thresholding):
    row, column, channel = img.shape
    list1 = [255]
    list2 = [0]
    for i in range(row):
        for j in range(column):
            if img[i,j,0] >= thresholding:
                list1.append(img[i,j,0])
            else:
                list2.append(img[i,j,0])
    list1.sort() # white
    list2.sort() # black
    sum1 = sum2 = 0
    for m in range(len(list1)):
        sum1 = sum1 + list1[m]
    for n in range(len(list2)):
        sum2 = sum2 + list2[n]
    avg1 = sum1/len(list1)
    avg2 = sum2/len(list2)
    newt = int((avg1+avg2) * 0.5)
    if abs(newt - thresholding) <= 1:
        return newt
    else:
        return calculateT(img,newt)


def binary_iterative(img):
    row, column, channel = img.shape
    thresholding = random.randint(0,255)
    newt = calculateT(img, thresholding)
    binary_thresholding(img, newt)

```
### Label Component

```python
def labelComponents(img):
    row, column, channel = img.shape
    Matrix = [[0 for x in range(column)] for y in range(row)]
    for i in range(row):
        for j in range(column):
            if img[i,j,0] == 255:
                Matrix[i][j] = 1
            else:
                Matrix[i][j] = 0

    label = 1
    list = [0,1]
    for m in range(row):
        for n in range(column):
            if Matrix[m][n] == 1:
                if Matrix[m-1][n] == 0 and Matrix[m][n-1] == 0:
                    label += 1
                    list.append(label)
                    Matrix[m][n] = label
                elif Matrix[m-1][n] != 0 and Matrix[m][n-1] == 0:
                    Matrix[m][n] = Matrix[m-1][n]
                elif Matrix[m-1][n] == 0 and Matrix[m][n-1] != 0:
                    Matrix[m][n] = Matrix[m][n-1]
                else:
                    if Matrix[m-1][n] == Matrix[m][n-1]:
                        Matrix[m][n] = Matrix[m][n-1]
                    elif Matrix[m-1][n] > Matrix[m][n-1]:
                        Matrix[m][n] = Matrix[m][n-1]
                        list[Matrix[m-1][n]] = Matrix[m][n-1]
                    else:
                        Matrix[m][n] = Matrix[m-1][n]
                        list[Matrix[m][n-1]] = Matrix[m-1][n]

    union(list)

    b = set(list)
    b1 = [k for k in b]
    numsofColor = len(b1)


    arr = []
    for x in range(numsofColor):
        count = random.randint(1,7)
        if count == 1:
            x = random.randint(0,255)
            arr.append([x,0,0])
        elif count == 2:
            x = random.randint(0,255)
            arr.append([0,x,0])
        elif count == 3:
            x = random.randint(0,255)
            arr.append([0,0,x])
        elif count == 4:
            x = random.randint(0,255)
            arr.append([x,x,0])
        elif count == 5:
            x = random.randint(0,255)
            arr.append([x,0,x])
        elif count == 6:
            x = random.randint(0,255)
            arr.append([0,x,x])
        else:
            x = random.randint(0,255)
            arr.append([x,x,x])

  # arr = types of color
  # b1 = types of labels

    for i in range(row):
        for j in range(column):
            for k in range(1,len(b1)):
                if list[Matrix[i][j]] == b1[k]:
                    img[i][j] = arr[k]
    return img


def find(i,list):
    if (i != list[i]):
        list[i] = find(list[i],list)
    return list[i]


def union(list):
    i = len(list)-1
    while i != 1:
        find(i,list)
        i  = i - 1
    return list
```
### Test Part

```python
# np.set_printoptions(threshold=np.inf)
img = mpimg.imread('/Users/gavin/Desktop/sample4.jpeg')
# plt.imshow(img, interpolation='nearest')
img.flags.writeable = True

# rgbToGray(img)
# smooth_avg(img)
# salt_pepper(img)
# smooth_median(img)
# binary_thresholding(img,180)
# binary_ptile(img,35)
# binary_iterative(img)
# smooth_median(img)
# smooth_median(img)
# smooth_median(img)
# smooth_median(img)
# histogram(img)
# labelComponents(img)
plt.imshow(img)
# plt.savefig('sample4.png')
plt.show()
```

# PyQt5

```python
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog
from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap, QImage

from test import *


class GUI(QMainWindow):

    def __init__(self, parent =None):
        super(GUI, self).__init__(parent)
        self.initUI()


    def initUI(self):
        self.resize(1500,500)
        self.setWindowTitle('imageProcessing')
        self.statusBar().showMessage('Author: Fei')
        self.horizontaol_vertical_box_layout()



    def horizontaol_vertical_box_layout(self):
        # label

        self.label_1 = QLabel('Input Image')
        self.label_1.setAlignment(Qt.AlignCenter)
        self.label_2 = QLabel('Output Image')
        self.label_2.setAlignment(Qt.AlignCenter)

        # button
        button_1 = QPushButton('The Original image')
        button_1.clicked.connect(self.btn1_clicked)

        button_2 = QPushButton('Use this image')
        button_2.clicked.connect(self.btn2_clicked)

        button_31 = QPushButton('Salt-Pepper')
        button_31.clicked.connect(self.btn31_clicked)
        button_32 = QPushButton('Avg Smooth')
        button_32.clicked.connect(self.btn32_clicked)
        button_33 = QPushButton('Median Smooth')
        button_33.clicked.connect(self.btn33_clicked)
        button_34 = QPushButton('T Binary')
        button_34.clicked.connect(self.btn34_clicked)
        button_35 = QPushButton('P-Tile Binary')
        button_35.clicked.connect(self.btn35_clicked)
        button_36 = QPushButton('Iterative Binary')
        button_36.clicked.connect(self.btn36_clicked)
        button_37 = QPushButton('Label Component')
        button_37.clicked.connect(self.btn37_clicked)

        vbox_1 = QVBoxLayout()
        vbox_1.addWidget(self.label_1)
        vbox_1.addWidget(button_1)

        vbox_2 = QVBoxLayout()
        vbox_2.addWidget(self.label_2)
        vbox_2.addWidget(button_2)

        vbox_3 = QVBoxLayout()
        vbox_3.addWidget(button_31)
        vbox_3.addWidget(button_32)
        vbox_3.addWidget(button_33)
        vbox_3.addWidget(button_34)
        vbox_3.addWidget(button_35)
        vbox_3.addWidget(button_36)
        vbox_3.addWidget(button_37)

        hbox_1 = QHBoxLayout()
        hbox_1.addLayout(vbox_1)
        hbox_1.addLayout(vbox_2)
        hbox_1.addLayout(vbox_3)

        layout_widget = QWidget()
        layout_widget.setLayout(hbox_1)

        self.setCentralWidget(layout_widget)


    # def btn1_clicked(self,checked):
    #     self.img1 = QFileDialog.getOpenFileName(self, 'OpenFile','.','Image Files(*.jpg *.jpeg *.png)')[0]
    #
    #     image1 = QImage(self.img1)
    #     self.label_1.setPixmap(QPixmap.fromImage(image1))



    def btn1_clicked(self,checked):
        self.img1 = mpimg.imread('/Users/gavin/Desktop/sample4.jpeg')
        self.img1.flags.writeable = True
        self.row, self.column, self.channel = self.img1.shape
        self.bytesPerLine = 3 * self.column
        qImg = QImage(self.img1.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_1.setPixmap(QPixmap.fromImage(qImg))
        plt.imshow(self.img1)
        # plt.savefig('sample4.png')
        plt.show()


    def btn31_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.salt_pepper(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def btn32_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.smooth_avg(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))


    def btn33_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.smooth_median(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def btn34_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.binary_thresholding(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def btn35_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.binary_ptile(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def btn36_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.binary_iterative(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))

    def btn37_clicked(self,checked):
        t = test(self.row, self.column, self.channel)
        self.img2 = t.labelComponents(self.img1)
        qImg = QImage(self.img2.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_2.setPixmap(QPixmap.fromImage(qImg))


    def btn2_clicked(self,checked):
        self.img1 = self.img2
        qImg = QImage(self.img1.data, self.column, self.row, self.bytesPerLine, QImage.Format_RGB888)
        self.label_1.setPixmap(QPixmap.fromImage(qImg))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())


```







