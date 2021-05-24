# 三维特征提取
该项目是在 [this code](https://github.com/kenshohara/3D-ResNets-PyTorch) 基础上修改而来的  
有两种模式:feature mode 和 score mode. 可以先在score mode下测试一下模型的有效性，然后再在feature mode下提取视频的三维特征。

## Requirements
* [PyTorch](http://pytorch.org/)   
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
* git clone 该项目代码
* 下载预训练的模型 [pretrained model](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
  * ResNeXt-101 表现最好. (See [paper](https://arxiv.org/abs/1711.09577) in details.)
  * 对于上述的3D-ResNeXt-101模型，如果输入的视频段的shape是(C, D, H, W)，经过多层conv和pooling，  
  输出的视频段shape将是(2048, D / 16, H / 32, W / 32). 但到最后由于有一个avgpooling操作，将输出一个  
  2048维的单一向量作为3D motion feature

## Usage
假设视频都放在 ```./videos```文件夹下.

采取 ```--mode score```，即可计算出视频中每一个分片的动作类别.
```
python main.py --video_dir $(your videos dir) --save_dir $(your save dir) --model $(checkpoints path) --mode score
```

为了计算每16帧的特征, 使用 ```--mode feature```.
```
python main.py --input ./input --video_root ./videos --output ./output.json --model ./resnet-34-kinetics.pth --mode feature
```


