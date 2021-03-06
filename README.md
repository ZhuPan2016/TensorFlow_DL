# TensorFlow_DL
本项目主要记录自己使用tensorflow学习深度学习的过程
### 2017-07-17 更新 conv_mnist.py  
##### 使用两层卷积层和两层池化层，实现对mnist数据集的分类
---
### 2017-07-27 更新 fully_connected_deep_network.py & fc_network_test.py
##### 使用TensorFlow读取csv文件数据
##### 创建两个隐藏层，每层200个结点，使用dropout随机失活，防止参数过多带来的过拟合
##### 感谢kevin28520 的一部分代码带给我的灵感
##### fc_network_test.py文件可以对生成的模型进行准确率的测试，并且生成分类的混淆矩阵
---
### 2017-08-05 更新 spectrum_conv/transform.py
##### 读取.wav音频文件，利用短时傅里叶变换生成频谱图
### 2017-08-31 更新 spectrum_conv/spec_cnn.py & test.py
##### 使用卷积神经网络对生成的频谱图进行分类
##### 对训练好的卷积神经网络在训练集上进行测试，查看识别率

识别结果：

![](https://raw.githubusercontent.com/ZhuPan2016/pic_bed/master/spec_conv_train.png)

相同的网络结构，将数据分成测试集和训练集进行测试
识别结果：

![](https://raw.githubusercontent.com/ZhuPan2016/pic_bed/master/spec_conv_tg_rain.png)

---
