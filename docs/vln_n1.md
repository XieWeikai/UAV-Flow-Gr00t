# VLN_N1

## 准备数据
1. 下载[InternData-n1](https://huggingface.co/datasets/InternRobotics/InternData-N1/tree/main/vln_ce/traj_data)
2. 解压其中的`vln_n1/traj_data`中的内容，目录结构组织为如下形式(以下只显示`hm3d_d435i`，实际有十二个目录，命名为`<scene>_<camera>`)
```bash
VLN-N1/hm3d_d435i
├── 00001-UVdNNRcVyV1__8b94c4e2ac
│   └── 00001-UVdNNRcVyV1
│       ├── data
│       ├── meta
│       └── videos
├── 00002-FxCkHAfgh7A__6741534e59
│   └── 00002-FxCkHAfgh7A
│       ├── data
│       ├── meta
│       └── videos
├── 00004-VqCaAuuoeWk__6464454dbb
│   └── 00004-VqCaAuuoeWk
│       ├── data
│       ├── meta
│       └── videos
...
```

## 数据转换
修改[vln_n1_simple.sh](scripts/vln_n1_simple.sh)中的`raw_dir`和`output_dir`路径为正确的路径，执行该脚本即可，各参数为：
- `num_processes` 启动多少个进程同时写入结果
- `raw_dir` 输入数据集路径，为`<scene>_<camera>`对应的路径
- `output_dir` 输出转换后的数据集路径
- `roll_limit` 限制筛选视角，比如为5.0则只会保留俯视<=5°的数据

