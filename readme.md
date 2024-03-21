![0](https://i.pinimg.com/originals/53/34/36/5334362c0a29f2603b3843587ada34bb.jpg) <br><br>



老師好、大家好，我是電子三丙的同學（108360721 陳靖元），<br>
這次的 Lab 我感到十足的有興趣，因為 Classification 需要使用大名鼎鼎的 CNN ，<br>
而且老師給予的題材又是我個人相當喜愛的辛普森家庭，<br>
所以我十分投入在本次實驗當中。<br>
<br>
以下是我的研究過程：
<br><br>

原理簡述
--
這次實驗主要是在探究卷積神經網路（Convolutional Neural Networks，CNN）的使用，<br>
CNN 能應用在圖形識別、圖形驗證；同時卷積神經網路也能應用在網路相關領域，<br>
比如網路加速器，算法應用加速器，能幫助人類用更快的方式，得到最好的結果。 <br><br>

CNN運算會去抓取不同大小的範圍去做重複比較，<br>
每一次被抓取出來的範圍就稱之為特徵（feature），對於以往的比較整張圖的像素陣列，<br>
抓取不同範圍的像素區域，能更好的來判斷圖形的相似處，<br>
如下圖：<br><br>

![image](https://d29g4g2dyqv443.cloudfront.net/sites/default/files/pictures/2018/convolutional_neural_network.png) <br><br>

原圖是一個較大的二維陣列，<br>
但藉由抓取局部的像素組成更多新的二維陣列，做比較運算，<br>
可以看到構成路牌符號的特徵就是中間的數字和周圍圓形，<br>
所以圖形有符合這些條件的話，在 CNN 運算面前就會被歸類於這個符號內。<br>

<br>
<br>

前置作業
--
<br>

```
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import scipy
import cv2
import imageio
import numpy
import h5py
%matplotlib inline

np.random.seed(2)

from sklearn.metrics import confusion_matrix
import itertools

from pathlib import Path
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import model_from_json
from keras.models import load_model

import warnings

warnings.filterwarnings('ignore')
```
<br>

以上是我使用到的函式庫，<br>
安裝方法請見我的上一偏 Lab 文章『Lab 1 Regression : House Sale Price Prediction Challenge』 <br>
連結：[https://github.com/MachineLearningNTUT/regression-ntut108360721](https://github.com/azppl2013/ML_Regression/blob/main/readme.md) <br><br>


主要程式撰寫
--
<br>

![image](https://user-images.githubusercontent.com/95005809/150534648-042c784b-9cdc-4b93-85da-711b66bbb1f1.png)

<br><br>
上圖確認圖片與對應的角色都有正確對應到後，<br>
才能開始撰寫主要的程式（我曾在這裡遇見問題，詳見下一章）。<br><br>

我的訓練架構是參考與比對多篇 Github 上的外國工程師，所得出的最佳解，<br>
架構如下：<br><br>

（[Conv2D->relu]*2 -> MaxPool2D -> Dropout）*2 -> Flatten -> Dense -> Dropout -> Out <br><br>

在這個學術研究的步驟常常花費太多時間，<br>
導致實做的部分被推延，<br>
學與做之間要取得平衡，對我來說還是一個值得探討的議題。<br><br>

![image](https://user-images.githubusercontent.com/95005809/150535439-c6ecf8b8-96d8-4e02-bda6-06a51bd0619b.png)<br>
![image](https://user-images.githubusercontent.com/95005809/150535804-d31f0553-e607-4ccb-b1cc-3f89ee8d6900.png)<br><br>

以上是訓練架構的部分，<br>
一開始由於硬體限制，訓練層數並沒有調整至上圖那麼高，<br>
運算架構與理論背景知識也尚未完整；<br>
所以訓練出來的網路如下圖，相當悽慘。<br><br>

![image](https://user-images.githubusercontent.com/95005809/150535873-3e1b4022-4f07-401d-8ee9-284d54b86408.png)<br><br>

但由於我的偏執，即使已經過了繳交期限，<br>
還是不願就此服輸，所以又拚了幾天，<br>
上網翻閱許多討論資料，以及 Github 上的工程師們的程式。<br>
將架構改為上述完整版，得到如下圖的最後結果。<br><br>

![image](https://user-images.githubusercontent.com/95005809/150536185-4b0831b6-3f1b-4534-b390-f232706390b4.png)<br><br>


撰寫程式中遇見的困難
--
<br>

這次遇到的碰撞並沒有如 Lab 1 那麼多，<br>
上次的實驗由於我是完全的 Python 新手，所以連建置環境都可以碰壁，<br>
可想而知結果並不盡人意。<br><br>

但這次我卯足了全力，想對於自己的能耐做出一些具有突破性的改進，<br>
想當然爾，僅有衝勁是不夠的，依然遇到了一些問題，<br>
於此我舉出一個使我最懊惱的難題（已解決）：<br><br>

在我寫完一個版本的程式後，丟進 Kaggle 訓練，<br>
得出的結果如下圖：<br><br>

![image](https://user-images.githubusercontent.com/95005809/150537177-7d704aca-4fef-48b3-bb71-a52e22a84f39.png)<br><br>

這個分數比全猜同一位腳色還低，甚至是第一版程式的十分之一，<br>
況且在第一時間並沒有找出問題，在努力好幾個小時後依然無果，<br>
這使我心灰意冷，最後好在老師有將修課同學們拉進同一個 Line 群。<br>
我在上面發問如下圖：<br><br>
![image](https://user-images.githubusercontent.com/95005809/150537660-961f91c7-bd9a-4274-a59c-a6ddc5b947f6.png)

<br><br>

最後熱心的同學/學長向我慷慨解囊，<br><br>

![image](https://user-images.githubusercontent.com/95005809/150538020-dbef14a0-e941-49e0-aa8b-9d55c940041e.png)

<br><br>


我順利地照著方法得到了這個模型應該有的結果，
![image](https://user-images.githubusercontent.com/95005809/150537870-dd74ea17-b296-4d48-abba-cd559e9772df.png)









