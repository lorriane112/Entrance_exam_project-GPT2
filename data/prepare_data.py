import json
import torch
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

class DataPreprocessor:
    def __init__(self, data_file):
        self.data_file = data_file
        
    def load_data(self):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def extract_text(self, data):
        """从JSON数据中提取文本内容"""
        texts = []
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # 假设您的JSON结构包含'content'或'text'字段
                    if 'content' in item:
                        texts.append(item['content'])
                    elif 'text' in item:
                        texts.append(item['text'])
                    elif 'story' in item:
                        texts.append(item['story'])
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(data, dict):
            # 遍历字典的所有值
            for value in data.values():
                if isinstance(value, str):
                    texts.append(value)
                elif isinstance(value, list):
                    texts.extend(self.extract_text(value))
        
        return texts
    
    def train_tokenizer(self, texts, vocab_size=30000):
        """训练BPE分词器"""
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )
        
        # 将文本写入临时文件供tokenizer训练
        temp_file = "temp_text.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
        
        tokenizer.train(files=[temp_file], trainer=trainer)
        os.remove(temp_file)
        
        return tokenizer
    
    def prepare_training_data(self, texts, tokenizer, block_size=256):
        """准备训练数据"""
        encoded_texts = []
        for text in texts:
            encoded = tokenizer.encode(text).ids
            encoded_texts.extend(encoded)
        
        # 转换为PyTorch tensor
        data = torch.tensor(encoded_texts, dtype=torch.long)
        
        # 保存分词器和数据
        tokenizer.save("data/tokenizer.json")
        torch.save(data, "data/train_data.pt")
        
        return data

def main():
    preprocessor = DataPreprocessor("data/train.json")
    data = preprocessor.load_data()
    texts = preprocessor.extract_text(data)
    
    print(f"提取了 {len(texts)} 个文本片段")
    print(f"总字符数: {sum(len(text) for text in texts)}")
    
    # 训练分词器
    tokenizer = preprocessor.train_tokenizer(texts)
    
    # 准备训练数据
    train_data = preprocessor.prepare_training_data(texts, tokenizer)
    
    print(f"训练数据形状: {train_data.shape}")
    print("数据预处理完成!")

if __name__ == "__main__":
    main()