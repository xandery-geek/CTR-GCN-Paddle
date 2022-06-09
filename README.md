# CTR-GCN-Paddle
> 该项目面向基于骨骼点的动作识别任务，是2021 CCF大数据与计算智能大赛（BDCI）- 基于飞桨实现花样滑冰选手骨骼点动作识别赛题。

## Usage
```
python main.py
```

## Task
基于给定的骨骼点数据集，对输入的人物动作进行分类。

骨骼点数据：
<div align=center>
  <image src="https://cdn.nlark.com/yuque/0/2021/png/805252/1632471962062-e2f52837-1833-49d2-a398-221fe62a6132.png" width=200px>  
</div>

## Dataset
比赛给出的数据集：所有视频素材均从2017-2020 年的花样滑冰锦标赛中采集得到。源视频素材中视频的帧率被统一标准化至每秒30 帧，图像大小被统一标准化至1080 * 720 ，以保证数据集的相对一致性。之后通过2D姿态估计算法Open Pose，对视频进行逐帧骨骼点提取，最后以.npy格式保存数据集。

train data: [N, C, T, V, M]

|维度符号表示|	维度值大小|	维度含义|	补充说明|
|---|---|---|---|
|N|	样本数|	代表N个样本|无|
|C|	3|	分别代表每个关节点的x,y坐标和置信度|每个x，y均被放缩至-1到1之间|
|T|	2500|	代表动作的持续时间长度，共有2500帧|	有的动作的实际长度可能不足2500，例如可能只有500的有效帧数，我们在其后重复补充0直到2500帧，来保证T维度的统一性|
|V|	25|	代表25个关节点|	具体关节点的含义可看下方的骨架示例图|
|M|	1|	代表1个运动员个数|	无|

## Result
- 在BDCI赛题的A榜上排名第8，B榜排名第16（模型的泛化能力不太好，也可能是调的参数还不太行）

## Model
模型采用的是 [**CTR-GCN** (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.html) 

## Improvement
- [x] 联合 Multi Stream 数据获得更好的预测结果，参考[Skeleton-Based Action Recognition With Multi-Stream Adaptive Graph Convolutional Networks](https://arxiv.org/abs/1912.06971)
- ~~CTR-GCN应该使用Transformation之后的结果进行channel-wise topology modeling~~
- ~~data preprocess: 删除无用的骨骼点~~
- Data Augmentation
  - 根据骨骼点的前后两帧计算中间帧得到新的数据，有以下两种计算方式:
  - [ ] average: 平均前后两帧的数据得到中间帧，`new_frame_data[i] = 0.5 * frame_data[i] + 0.5 frame_data[i+1]`
  - [x] confidence: 根据置信度对前后两帧的数据进行加权，然后计算中间帧，`new_frame_data[i] = confidence[i] * frame_data[i] + confidence[i+1] * frame_data[i+1]`
- [ ] Focal loss：由于赛题给出的样本不均衡，所以训练样本少的类别的正确率很低。尝试使用Focal loss改进，但是没Train出来。[Focal loss](https://arxiv.org/abs/1708.02002)
- 结合Transformer的思想(还未尝试)
  - [ ] Temporal : 1D CNN -> Transformer
  - [ ] Graph Structure: GCN -> Transformer
  
