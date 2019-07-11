# Learning-Rich-Features-for-Image-Manipulation-Detection
基于双流 Faster R-CNN 网络的 图像篡改检测

# 代码说明

本人学习能力和资源实在有限，所以本实验主要是对**[dBeker](https://github.com/dBeker)**的**Faster-RCNN-TensorFlow-Python3**GitHub仓库代码进行学习和一定的修改，从而实现了双流篡改检测。

参考链接：

https://github.com/dBeker/Faster-RCNN-TensorFlow-Python3

## 部署说明

首先修改`_lib`文件夹为`lib`。

由于GitHub文件上传大小的限制，预处理网络和训练好的模型请从网盘下载：
链接: https://pan.baidu.com/s/1eav5wfVrHFzeP1xCsp_fsQ 提取码: pm8m 

分享包括两个文件夹：
* vgg16网络
* 训练好的网络参数

将vgg16网络文件夹下的`.ckpt`文件放在`Learning-Rich-Features-for-Image-Manipulation-Detection/data/imagenet_weights/`文件夹下；
将训练好的网络参数文件夹下的四个文件放在`Learning-Rich-Features-for-Image-Manipulation-Detection/default/gene_2007_trainval/default/`文件夹下。


然后直接运行`双流Faster R-CNN.ipynb`文件。

### 运行异常处理

- 生成数据集时，如果出现：

  ```python
  ModuleNotFoundError: No module named 'tensorflow'
  ```

  需要下载`tensorflow`模块，在命令行输入：

  ```python
  pip install tensorflow
  ```

- 缺失cython_bbox函数

  需要对你所在环境下重新编译bbox.c，生成cython_bbox.so文件，在命令行执行：

  ```python
  cd ./lib
  python setup.py build_ext -i
  ```

