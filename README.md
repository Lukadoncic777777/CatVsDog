晚饭后在 CSDN 上看到了一个猫狗对战的项目，感觉很有意思，遂复现。

实现如下功能：

使用 torch.nn 训练神经网络，训练集为 kaggle 上的数据集，划分其中一部分为测试集，并导出模型，在 predict.py 中输入想预测照片的路径，预测其是狗还是猫。

使用 anaconda 创建虚拟环境，安装 python 3.12。并安装 torch 、 scikit-learn 和 opencv-python 。

运行 Main.py ,argv1 为 0/1, argv2 为图片集路径，argv3 为如果 argv1 为 1，则输入已训练好模型的路径。
