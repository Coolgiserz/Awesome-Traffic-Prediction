# Graph WaveNet

## Paper

- [IJCAI 2019] Graph WaveNet for Deep Spatial-Temporal Graph Modeling  
  Paper: https://www.ijcai.org/Proceedings/2019/0264.pdf  
  Code: https://github.com/nnzhan/Graph-WaveNet

## Why It Matters

Graph WaveNet is a representative traffic forecasting baseline that combines:

- Dilated temporal convolutions for long-range temporal dependencies.
- Adaptive adjacency learning for hidden spatial relations between sensors.
- End-to-end training on common benchmarks such as METR-LA and PEMS-BAY.

## Typical Usage Notes

- Input: historical multivariate traffic signals on graph nodes.
- Output: multi-horizon forecasts (for example 15/30/60 min ahead).
- Common metrics: MAE, RMSE, MAPE.

## Reproducibility Checklist

- Report exact data split and preprocessing.
- Report prediction horizons and masking rules.
- Report runtime environment and random seeds.

## 中文说明

Graph WaveNet 是交通预测中常用的经典基线之一，核心在于结合时序卷积与自适应图结构学习。

- 通过膨胀时序卷积建模长时依赖。
- 通过自适应邻接矩阵补充显式路网之外的隐式空间关系。
- 常用于 METR-LA、PEMS-BAY 等基准。
