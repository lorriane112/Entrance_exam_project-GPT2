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

4.Result
最终会生成10个文字样本，存储在continued_article.txt里

第12章 替嫁（十二）
“什么，你们就要走了？”
凌妙妙的嘴张得老大，“明日就动身，这……这么急吗？”
话音未落，脑海里重重叠叠响起数声警告的“叮”声，宛如冲垮了堤坝的洪水，一股脑儿地奔涌而出。
不用听也知道，她的任务完成度太低，现在主角团都要离开太仓了，别说慕声那边没一点起色，就连与柳拂衣的亲密度也没刷够。
“凌小姐，”慕瑶难见地给了她一个温柔的微笑，“捉妖人以四海为家，以漂泊为命，我们在这里已经叨扰太久了。”
她的眼中有一种潇洒的神采，尤其是说到“四海为家”的时候，声音清凌凌的，掷地有声，就像个仗剑天涯的女侠。]

==================================================
【续写部分】
==================================================
漂泊为命 ， 我们在这里已经叨扰太久了 。” 她的眼中有一种潇洒的神采 ， 尤其是说到 “ 四海为家 ” 的时候 ，
声音清凌凌的 ， 掷地有声 ， 就像个仗剑天涯的女侠 。] ， 掷地有声 ， 就像个仗剑天涯的女侠 ，“
仍然精力充沛 、 热情似火 。 这种发疯一般都兴奋显然也感染了慕声 ， 他仅有的几丝睡意也烟消云散了 。 
“ 凌虞 。” 慕声也开始叫她 。 “ 别叫我凌虞 。” 妙妙垮下脸 ，“ 难听 。”
凌虞 ， 可不就是囹圄 ， 困了原身一辈子 ？ 
慕声完全抛弃了自己礼貌的假面 ， 抬抬眼皮 ：“‘ 凌小姐 ’ 三个字 ， 拗口 。”