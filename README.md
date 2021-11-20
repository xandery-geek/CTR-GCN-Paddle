# CTR-GCN-Paddle

## improve
- [x] Multi Stream 联合训练
- [ ] ~~CTR-GCN应该使用Transformation之后的结果进行channel-wise topology modeling~~
- 结合Transformer的思想
  - Temporal : 1D CNN -> Transformer
  - Graph Structure: GCN -> Transformer
~~- data preprocess: 删除无用的骨骼点~~

~~- Unsupervised~~
  - 伪标签
- Trick
  - data augmentation
    - [ ] average
    - [x] confidence
  - TTA（测试时增强）