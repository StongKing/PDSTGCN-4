# 预测结果与再平衡测试数据

运行 `export_rebalancing_data.py` 会读取当前最佳轮次对应的测试预测文件，
核对节点、样本顺序、原始库存时间戳，并用同一权重生成验证集预测。
不训练、不改模型、不覆盖原预测和权重。

## 直接运行

在项目的 `pytorch_gpu39` 环境运行：

```powershell
& 'D:\anaconda3\envs\pytorch_gpu39\python.exe' 'D:\PDSTGCN-3\export_rebalancing_data.py'
```

第一次需补算验证集，后续复用输出目录中的缓存。默认读取 `best_epoch.txt`
指定轮次的 `output_epoch_<轮次>_test.npz` 与 `epoch_<轮次>.params`，
并验证该权重能够复现测试文件的首个批次。不要把不同轮次的预测和权重混用。

本次数据在：
`outputs/01a0a96b-0035-7f22-b360-8fc2b56e2ddf/rebalancing_epoch_86/`。

## 文件用途

| 文件 | 内容 |
|---|---|
| `再平衡数据筛选.xlsx` | 本次生成的可筛选工作簿，含示例的全部站点、场景索引、鲁棒区间和站点表现 |
| `scenario_index.csv` | 5346 行，每行是一个预测起点与一个预测步，含 sample_id、graph_id、起点和目标时间 |
| `all_nodes_10min.csv.gz` … `all_nodes_60min.csv.gz` | 每个文件 506088 行，含在途节点与全部站点、库存、误差、经纬度；可解压后在 Excel 打开 |
| `all_predictions.npz` | 全量数组，prediction/actual/error 维度为 `[891,568,6]`；current_inventory 为 `[891,568]` |
| `robust_bounds_validation.csv` | 由验证集校准的在途误差及全部站点误差和的上下界、测试覆盖率 |
| `station_bounds_validation.csv` | 每个节点、每个预测步的局部误差分位数与极值 |
| `station_metrics_test.csv` | 各节点、各预测步的测试 MAE、RMSE、偏差和误差分位数；属于事后统计 |
| `calibration_errors.npz` | 保留全部节点的成对验证残差，便于为任意固定站点集合重新校准 |
| `case_s0080_h60/selected_stations.csv` | 默认 60 分钟、10 个站点示例的数据 |
| `case_*/all_stations.csv` | 当前示例全部567个真实站点，selected=1表示选入 |
| `case_*/target_inventory_template.csv` | 待填写的目标库存，空值不等于0 |
| `case_*/paired_calibration_errors.csv` | 在途误差、选中站点误差和、未选站点误差和，每行来自同一个验证场景 |
| `case_*/case_parameters.json` | 时刻、站点ID、子集区间、全系统区间和符号说明 |
| `case_*/straight_line_distance_km.csv` | 所选站点间球面直线距离，不是道路距离或行驶时间；未指定仓库 |
| `manifest.json` | 原预测、权重指纹，校准样本数，误差及总量核验结果 |

所有 CSV 使用 UTF-8 BOM，便于中文 Windows Excel 读取。站点ID来自原数据；
`node_index` 是模型数组位置，二者不应混淆。`sample_id` 从0开始。
库存单位为辆，时间保留源数据的钟面时间，不转换为电脑所在地时区。
这里的 actual 是 V7 数据集中的重构标签，不是新采集的实时站点观测。

## 挑选其他场景与站点

先筛选 `scenario_index.csv` 的 sample_id，再运行，例如：

```powershell
& 'D:\anaconda3\envs\pytorch_gpu39\python.exe' 'D:\PDSTGCN-3\export_rebalancing_data.py' --sample-id 80 --horizon-minutes 30 --station-count 20 --skip-full-csv
```

明确指定原始站点ID：

```powershell
& 'D:\anaconda3\envs\pytorch_gpu39\python.exe' 'D:\PDSTGCN-3\export_rebalancing_data.py' --sample-id 80 --horizon-minutes 60 --station-ids 100,287,43,37,47,51,52,286,13,21 --skip-full-csv
```

不指定 sample_id 时选测试集中第一个08:00起点；不指定站点ID时按该场景
预测库存相对当前库存的绝对变化排序。它只是一份格式示例，没有要求站点紧邻，
也没有根据测试真实误差挑选“表现最好”的案例。对路线测试可自行指定地理相邻站点。

Python 脚本每次重建 CSV/NPZ/JSON，不自动重建 Excel。若本机具有当前 Codex 表格运行环境，
更新工作簿可运行下面的独立构建器；第二个参数可指定其他导出目录：

```powershell
& 'C:\Users\JZS\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' 'D:\PDSTGCN-3\outputs\01a0a96b-0035-7f22-b360-8fc2b56e2ddf\build_rebalancing_workbook.mjs'
```

Excel 中修改选用列会更新站点数量和所选名义余量。改变站点集合后，应通过
`--station-ids` 重新运行 Python 脚本以校准新的子集区间；表中显示的全系统区间不随子集改变。
Excel 的手工填写不会自动传回 CSV。目标库存请在 CSV 模板中填写后用 `--target-inventory <路径>` 导入。
重新构建工作簿会替换工作簿中的手工修改，应先保存另一个副本或导出填写的 CSV。

## 误差符号与守恒关系

统一使用 `error_actual_minus_prediction = actual_inventory - predicted_inventory`。
正值表示模型低估库存，负值表示模型高估库存。

若真实值和预测值均满足全系统总量5310，令 Δ_i=实际_i−预测_i，则：

```
Σ(全部真实站点 Δ_i) = -Δ_0
```

若验证集在途误差范围为 `[L0,U0]`，应对全部站点施加：

```
-U0 <= Σ Δ_i <= -L0
```

不要漏掉符号翻转及上下界交换。不要求区间对称。
输出同时提供经验极值 `[min,max]` 和中央95%区间 `[q2.5%,q97.5%]`。
若需要整数区间的保守外包，可对下界向下取整、上界向上取整。
历史极值不是未来的确定性保证。这里的95%是每个预测步的经验边际区间，
不是567个站点或6个预测步同时落入区间的95%保证。

若只调度集合 S，保留未选站点的合计误差 Δ_other：

```
Σ(i属于S) Δ_i + Δ_other + Δ_0 = 0
```

脚本为固定示例集合直接从成对验证残差校准子集和的区间，也输出完整成对残差。
不能把全系统在途界原样套到任意子集，除非还明确处理未选站点。
用不同分位区间分别约束三个量时，不应把它们误称为联合95%保证。

## 从库存到再平衡量

`predicted_change_from_current` 只是未来预测库存相对当前库存的变化，
不是调度量。目标库存 e_i 必须根据你的调度模型自行给定。

```
nominal_surplus = predicted_inventory - target_inventory
required_addition = target_inventory - predicted_inventory
```

目标库存留空时以上两项留空，绝不默认成0。与论文 d_i=p_i-e_i 定义一致时，
名义余量为 d_i；实际 d_i 的不确定偏差仍是 Δ_i=actual−prediction。
工作簿中实际库存与实际误差只用于事后评估，不应输入当时的决策过程。

## 校准及当前实验范围

区间全部使用同一 epoch 86 权重生成的验证预测校准；不从测试标签拟合上下界。
移除了5个目标结束时刻晚于首个测试决策时刻的验证窗口，剩余886个。
但该验证集仍曾用于权重选择，窗口间也相关，因此这些是用于流程测试的经验区间。
后续正式比较可独立划分训练、选模、校准和测试时段，并按时段检验覆盖率。

本导出保留当前实验，不修正上一轮已确认的动态图多包含一个未来区间的问题。
当前数据可用于搭建再平衡流程；正式报告预测与调度优势前，需要修正时间对齐后重跑。

## Python 直接读取

```python
import numpy as np
import pandas as pd
from pathlib import Path

folder = Path(r"D:\PDSTGCN-3\outputs\01a0a96b-0035-7f22-b360-8fc2b56e2ddf\rebalancing_epoch_86")
z = np.load(folder / "all_predictions.npz", allow_pickle=False)
prediction_60min = z["prediction"][80, 1:, 5]  # sample=80, exclude node0, horizon=60min
actual_60min = z["actual"][80, 1:, 5]
current = z["current_inventory"][80, 1:]
stations = pd.read_csv(folder / "case_s0080_h60/selected_stations.csv", dtype={"node_id": str})
robust = pd.read_csv(folder / "robust_bounds_validation.csv")
```
