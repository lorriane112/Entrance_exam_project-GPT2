#!/bin/bash

echo "开始数据预处理..."
python data/prepare_data.py

echo "开始训练模型..."
python model/train.py

echo "训练完成!"