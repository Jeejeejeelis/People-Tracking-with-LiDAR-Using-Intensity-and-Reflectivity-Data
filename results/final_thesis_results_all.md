| Architecture | Input Features | Optimal Tracker Heuristic | AMOTA | HOTA | DetA | AssA | LocA | IDF1 |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HMFN** | Baseline (5-feat: $I$) | Untuned | **71.7%** | 14.3% | 36.2% | 5.6% | 89.3% | 10.1% |
| **HMFN** | Decoupled (6-feat: $I, d$) | Untuned | **68.9%** | 13.4% | 33.8% | 5.3% | 89.3% | 9.2% |
| **HMFN** | Kitchen Sink (7-feat: $I, R, d$) | Untuned | **68.0%** | 13.9% | 35.9% | 5.4% | 88.2% | 9.8% |
| **HMFN** | Swap (6-feat: $R, d$) | Tuned | **36.8%** | 28.9% | 46.2% | 18.1% | 84.0% | 28.1% |
| **CenterPoint** | Baseline (5-feat: $I$) | Untuned | **0.0%** | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| **CenterPoint** | Decoupled (6-feat: $I, d$) | Untuned | **70.3%** | 17.9% | 39.8% | 8.1% | 77.8% | 16.8% |
| **CenterPoint** | Kitchen Sink (7-feat: $I, R, d$) | Untuned | **68.2%** | 25.3% | 52.1% | 12.3% | 87.0% | 22.7% |
| **CenterPoint** | Swap (6-feat: $R, d$) | Tuned | **68.6%** | 38.6% | 67.2% | 22.2% | 87.4% | 37.3% |
| **VoxelNeXt** | Baseline (5-feat: $I$) | Untuned | **38.5%** | 14.9% | 39.3% | 5.7% | 76.0% | 11.5% |
| **VoxelNeXt** | Decoupled (6-feat: $I, d$) | Untuned | **70.3%** | 21.7% | 43.4% | 10.8% | 88.0% | 18.7% |
| **VoxelNeXt** | Kitchen Sink (7-feat: $I, R, d$) | Untuned | **58.7%** | 21.6% | 44.7% | 10.5% | 87.9% | 18.6% |
| **VoxelNeXt** | Swap (6-feat: $R, d$) | Tuned | **70.1%** | 30.5% | 56.5% | 16.5% | 87.5% | 30.0% |
| **PointPillars** | Swap (6-feat: $R, d$) | Tuned | **0.0%** | 17.9% | 30.3% | 10.6% | 69.0% | 15.1% |
