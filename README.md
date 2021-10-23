# CTR-GCN-Paddle

## improve
- Multi Stream 联合训练，而不是单独训练再加权预测
- CTR-GCN应该使用Transformation之后的结果进行channel-wise topology modeling
- 结合Transformer的思想
  - Temporal : 1D CNN -> Transformer
  - Graph Structure: GCN -> Transformer
  - 