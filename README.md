# UAVFlow2LeRobot
该仓库用于将UAVFlow数据集转换成[gr00t](https://github.com/NVIDIA/Isaac-GR00T)可用的LeRobot数据集，同时提供使用UAVFlow数据集（LeRobot格式）训练Gr00t模型所需的相关文件。

## 环境配置
本仓库使用`uv`管理python环境
```bash
uv sync
# 之后使用uv run ...即可
# 该命令会在当前目录创建.venv并安装所需的依赖
```

---
## 数据集转换
下载UAV-Flow数据集，该数据集分为[真实数据](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow/tree/main)和[Sim数据](https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow-Sim/tree/main)，数据的目录结构如下所示
```bash
UAV-Flow
├── train-00000-of-00054.parquet
├── train-00001-of-00054.parquet
...
└── train-00053-of-00054.parquet


UAV-Flow-Sim/
├── train-00000-of-00021.parquet
├── train-00001-of-00021.parquet
...
└── train-00020-of-00021.parquet
```
在项目根目录下使用如下命令转换数据集为LeRobot格式
```bash
python uav_flow_2_lerobot.py --repo-id UAVFlowLeRobot --data-path path/to/UAV-Flow --eval-trajectories 200
```
该命令会生成两个目录`<repo-id>-train`和`<repo-id>-eval`，在这个例子中即`UAVFlowLeRobot-train`和`UAVFlowLeRobot-eval`，这两目录即为LeRobot v2格式的数据（LeRobot已经出v3版本了，Gr00t是自己对LeRobot v2做了支持，并没有直接用LeRobot的官方实现）。

---
## 训练并推理Gr00t
1. clone Gr00t官方仓库
```bash
cd <some-path> # 到该项目所在的目录下
git clone git@github.com:NVIDIA/Isaac-GR00T.git # or http
```
2. 下载(Gr00t)[https://huggingface.co/nvidia/GR00T-N1.5-3B]模型
```bash
# hf是huggingface新的cli名称
# 老版本叫huggingface-cli
hf download --repo-type model --local-dir path/to/GR00T-N1.5-3B nvidia/GR00T-N1.5-3B
```
3. 将相关文件复制到Gr00t项目中
```bash
cp -r gr00t_example/UAV_Flow path/to/Isaac-GR00T/examples
# 复制训练和推理脚本
cp gr00t_example/*.sh path/to/Isaac-GR00T

cd path/to/Isaac-GR00T
# 注意，Gr00t的数据集特有文件，需要在meta下放一个modality.json
# 没有该文件Gr00t自己的数据集代码无法工作
# path/to/data即之前转换后的LeRobot数据集的路径
cp examples/UAV_Flow/modality.json path/to/data/meta
```

4. 训练+推理，注意需要自行配置好GR00T的环境
```bash
cd path/to/Isaac-GR00T

# 开启训练
# 注意要修改其中的一些参数
./train_uav_flow.sh

# 启动推理服务
#注意修改其中一些参数
./inference_uav_flow.sh
```

