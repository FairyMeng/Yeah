# Does Noise in the Knowledge Graph Really Harm Recommendations?

This is the code of the *Does Noise in the Knowledge Graph Really Harm Recommendations?* in Pattern Recognition.

## Environment

python 3.6

Pytorch 0.3.x

visdom if visualization flag is set to True.

some required packages are included in *requirements.txt*.

## General Flags

We specify the general parameters by setting flags like optimizer or learning rate, which can be found in './models/base.py'.

## run
`python run_knowledgable_recommendation.py -model_type yeah -dataset ml1m -data_path /Users/yeahm/Documents/code/PR-Yeah/datasets/ -log_path /Users/yeahm/Documents/code/PR-Yeah/log/ -rec_test_files valid.dat:test.dat -nohas_visualization`

## 项目结构

```
Yeah_Rec/
├── models/
│   ├── base.py                 # 基础模型配置
│   ├── knowledgable_recommendation.py  # 主要模型实现
│   ├── yeah.py                 # yeah 模型实现
│   ├── transH.py               # TransH 模型
│   └── transUP.py              # TransUP 模型
├── data/
│   └── load_kg_rating_data.py  # 数据加载
└── utils/
    ├── trainer.py              # 训练器
    ├── misc.py                 # 工具函数
    ├── loss.py                 # 损失函数
    └── data.py                 # 数据处理
```

## 数据集格式

数据集应包含以下文件：
- train.dat
- valid.dat
- test.dat
- u_map.dat
- i_map.dat
- kg/
  - train.dat
  - valid.dat
  - test.dat
  - e_map.dat
  - r_map.dat