   ## 项目概述

在这一项目中，给你一个猫狗的图像数据集，训练模型并识别猫狗。本项目是来自kaggle中的猫狗识别项目。
我们通过深度学习的方法来设计一个识别猫狗的深度网络模型。在kaggle上的得分排名进入到前20名，相当于
取得了%2的名次，达到了商用的级别。并且提供了相关论文来详细说明整个算法的设计。




## 项目指南

### 步骤
1. 下载[猫狗数据集](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) 



### 安装必要的 Python 依赖包


	对于 __Mac/OSX__：
	
	```bash
	conda env create -f requirements/dogcat-mac.yml
	source activate dogcat-project
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```

	对于 __Windows__：
	
	```bash
	conda env create -f requirements/dogcat-windows.yml
	activate dogcat-project
	set KERAS_BACKEND=tensorflow
	python -c "from keras import backend"
	```
	
	对于 Linux：：
	
	```bash
	conda env create -f requirements/dogcat-linux.yml
	source activate dogcat-project
	KERAS_BACKEND=tensorflow python -c "from keras import backend"
	```
  **如果要安装gpu版本的话,那么替换相应的gpu版本的yml文件**
  
  
 ### 项目详细设计:
 请看猫狗识别论文.pdf,里面有详细的算法设计说明和实验数据分析。
 
