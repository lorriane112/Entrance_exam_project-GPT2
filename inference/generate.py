import json
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nanogpt_model import GPT
from config.model_config import GPTConfig
from tokenizers import Tokenizer

class ArticleContinuer:
    def __init__(self, model_path='model_best.pth'):
        print("📖 初始化文章续写器...")
        
        # 加载分词器
        self.tokenizer = Tokenizer.from_file("data/tokenizer.json")
        vocab_size = self.tokenizer.get_vocab_size()
        
        # 加载模型
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=256,
            n_layer=12,
            n_head=12,
            n_embd=768
        )
        
        self.model = GPT(config)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()
        print("✅ 模型加载完成")
    
    def extract_full_article(self, json_data):
        """从JSON中提取完整的文章内容"""
        full_text = ""
        
        def extract_text(obj):
            nonlocal full_text
            if isinstance(obj, str):
                full_text += obj + "\n"
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_text(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text(item)
        
        extract_text(json_data)
        return full_text.strip()
    
    def continue_article_end(self, article_text, num_sentences=10, context_length=500):
        """在文章末尾续写指定数量的句子"""
        print(f"📝 原文长度: {len(article_text)} 字符")
        
        # 取文章最后部分作为上下文
        if len(article_text) > context_length:
            context = article_text[-context_length:]
            print(f"使用最后 {context_length} 字符作为上下文")
        else:
            context = article_text
            print(f"使用全文作为上下文")
        
        print(f"上下文内容: ...{context[-100:]}")
        
        # 准备输入
        input_ids = torch.tensor([self.tokenizer.encode(context).ids])
        generated_ids = input_ids.clone()
        
        print(f"\n🎯 开始续写 {num_sentences} 个句子...")
        
        sentences_generated = 0
        continuation_text = ""
        
        with torch.no_grad():
            while sentences_generated < num_sentences:
                # 确保输入不超过模型限制
                if generated_ids.size(1) >= self.model.config.block_size:
                    generated_ids = generated_ids[:, -self.model.config.block_size:]
                
                # 前向传播
                logits, _ = self.model(generated_ids)
                next_token_logits = logits[:, -1, :] / 0.8  # temperature=0.8
                
                # Top-k 采样
                top_k = 50
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
                
                # 采样下一个token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 添加到生成序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 解码当前生成的内容
                current_full = self.tokenizer.decode(generated_ids[0].tolist())
                current_continuation = current_full[len(context):]
                
                # 检查是否生成了新的句子（中文句子结束符）
                sentence_endings = ['。', '！', '？', '……']
                if any(marker in current_continuation for marker in sentence_endings):
                    # 统计句子数量
                    new_sentences = 0
                    for marker in sentence_endings:
                        new_sentences += current_continuation.count(marker)
                    
                    if new_sentences > sentences_generated:
                        sentences_generated = new_sentences
                        continuation_text = current_continuation
                        print(f"✅ 已生成 {sentences_generated}/{num_sentences} 个句子")
                
                # 安全停止：如果生成太长但句子不够
                if generated_ids.size(1) - input_ids.size(1) > 500:  # 最多生成500个token
                    print("⚠️  达到生成长度限制，提前停止")
                    break
        
        return continuation_text
    
    def save_continued_article(self, original_text, continuation, output_file):
        """保存续写后的完整文章"""
        full_article = original_text + "\n\n" + "="*50 + "\n【续写部分】\n" + "="*50 + "\n" + continuation
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_article)
        
        # 同时保存JSON格式的结果
        result_data = {
            "original_article": original_text,
            "continuation": continuation,
            "full_article": original_text + continuation
        }
        
        json_output_file = output_file.replace('.txt', '.json')
        with open(json_output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        return full_article

def main():
    # 初始化续写器
    continuer = ArticleContinuer()
    
    # 您的JSON文件路径
    json_file = "data/train.json" 
    
    if not os.path.exists(json_file):
        print(f"❌ 文件不存在: {json_file}")
        print("请将您的JSON文件放在项目根目录，并修改脚本中的文件路径")
        return
    
    # 读取JSON文件
    print(f"📚 读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 提取完整文章
    article_text = continuer.extract_full_article(data)
    
    if not article_text:
        print("❌ 未能从JSON中提取到文本内容")
        return
    
    print(f"📖 提取到文章，长度: {len(article_text)} 字符")
    print(f"文章结尾: ...{article_text[-100:]}")
    
    # 在文章末尾续写10个句子
    print("\n" + "="*60)
    print("🚀 开始续写文章结尾...")
    print("="*60)
    
    continuation = continuer.continue_article_end(
        article_text, 
        num_sentences=10,
        context_length=400  # 使用最后400字符作为上下文
    )
    
    # 显示结果
    print("\n" + "="*60)
    print("🎉 续写完成！")
    print("="*60)
    
    print(f"\n📖 原文结尾:")
    print(f"...{article_text[-200:]}")
    
    print(f"\n✨ 续写的10个句子:")
    print(continuation)
    
    # 保存结果
    output_file = "continued_article.txt"
    full_article = continuer.save_continued_article(article_text, continuation, output_file)
    
    print(f"\n💾 结果已保存到:")
    print(f"  - {output_file} (文本格式)")
    print(f"  - {output_file.replace('.txt', '.json')} (JSON格式)")
    
    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"  原文长度: {len(article_text)} 字符")
    print(f"  续写长度: {len(continuation)} 字符")
    print(f"  续写句子数: {continuation.count('。') + continuation.count('！') + continuation.count('？')} 个")

if __name__ == "__main__":
    main()