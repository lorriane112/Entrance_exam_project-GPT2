# Entrance_exam_project-GPT2
1.Description:
从头训练一个82M的中文GPT2模型，使用BERT中文分词器采用《永夜星河》小说的部分章节。最终续写了10句的小说。

2.structure
Entrance_exam_project-GPT2/
├── config/
│   ├── train_config.py          # 训练配置
│   └── model_config.py          # 模型配置
├── data/
│   ├── clean_data.py            # 对原始数据集文件进行处理，去除其中非法字符，生成预处理好的数据集文件train.json。
│   ├── prepare_data.py          # 数据预处理脚本
│   ├── input.txt                # 小说原始文本
│   ├── txt_2_json.py            # 将input.txt文本处理为input.json
│   ├── train.json               # 处理好的input.txt
│   ├── tokenizer.json           # BERT分词器
│   └── train_data.pt            # 预处理训练数据
├── model/
│   ├── nanogpt_model.py         # GPT-2模型实现
│   └── train.py                 # 训练脚本
├── inference/
│   └── generate.py        # 小说续写推理
├── utils/
│   └── helpers.py               # 工具函数
├── model_best.pth               # 最佳模型权重文件
├── model_final.pth              # 最终模型权重文件
├── continued_article.txt        # 小说续写结果文件
├── requirements.txt             # 依赖包
├── train.sh                    # 训练脚本
└── README.md                   # 项目说明

3.start
(1)environment

首先下载依赖
pip install -r requirements.txt

(2)处理文本

分别运行
python txt_2_json.py
python clean_data.py  
python prepare_data.py  

(3)Training

bash train.sh
python model/train.py

(4)文本生成

bash inference.sh