# VLN_CE
本仓库并没有直接使用VLN_CE R2R RxR的原始数据，而是使用了[InternData-n1](https://huggingface.co/datasets/InternRobotics/InternData-N1/tree/main/vln_ce/traj_data)中处理过的VLN-CE数据集，好处是转换后可以直接用于训练而不需要从Habitat中自行收集数据，坏处是数据固定死了没那么灵活（原始数据应该是标注了轨迹，需要自己从habitat中取传感器数据，可以自行设置相机个数/参数/位置，所有Habitat相关的都可以自行调整）。

## 准备数据
1. 下载[InternData-n1](https://huggingface.co/datasets/InternRobotics/InternData-N1/tree/main/vln_ce/traj_data)
2. 解压其中的`vln_ce/traj_data`，目录结构组织为如下形式(以下只显示`rxr`，实际会有`rxr`，`r2r`，`scalevln`三个并列的目录)
```bash
VLN-CE/rxr
├── 17DRP5sb8fy__02be9c2011
│   └── 17DRP5sb8fy
│       ├── data
│       ├── meta
│       └── videos
├── 1LXtFkjw3qL__95f93db5b8
│   └── 1LXtFkjw3qL
│       ├── data
│       ├── meta
│       └── videos
├── 1pXnuDYAj8r__d4bb2d3a8d
│   └── 1pXnuDYAj8r
│       ├── data
│       ├── meta
│       └── videos
...
```

## 数据转换
修改[vln_ce_simple.sh](docs/vln_ce.md)中的`raw_dir`和`output_dir`路径为正确的路径，执行该脚本即可，生成的数据形如(以r2r为例)：
```bash
./VLN-CE-r2r
├── data
│   ├── chunk-000
│   ├── chunk-001
│   ├── chunk-002
│   ├── chunk-003
│   ├── chunk-004
│   ├── chunk-005
│   ├── chunk-006
│   ├── chunk-007
│   ├── chunk-008
│   ├── chunk-009
│   └── chunk-010
├── meta
└── videos
```
