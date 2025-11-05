import sys
import os
import torch
import time
import math
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.nanogpt_model import GPT
from config.train_config import TrainConfig
from config.model_config import GPTConfig
from utils.helpers import get_batch

class nullcontext:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

def main():
    config = TrainConfig()
    
    # 设置精度
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config.dtype]
    ctx = torch.amp.autocast(device_type='cuda', dtype=ptdtype) if config.device == 'cuda' else nullcontext()
    
    # 检查数据文件是否存在
    if not os.path.exists('data/train_data.pt'):
        print("错误: 训练数据文件不存在，请先运行数据预处理!")
        return
    
    # 加载数据
    train_data = torch.load('data/train_data.pt')
    print(f"训练数据加载完成，总tokens: {len(train_data)}")
    
    # 计算词汇表大小
    from tokenizers import Tokenizer
    if not os.path.exists('data/tokenizer.json'):
        print("错误: 分词器文件不存在!")
        return
        
    tokenizer = Tokenizer.from_file("data/tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")
    
    # 初始化模型
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    
    model = GPT(model_config)
    model.to(config.device)
    
    # 打印参数数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {n_params:,}")
    
    # 优化器
    optimizer = model.configure_optimizers(
        weight_decay=0.1,
        learning_rate=config.learning_rate,
        betas=(0.9, 0.95)
    )
    
    # 学习率调度器
    def get_lr(it):
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        if it > config.lr_decay_iters:
            return config.min_lr
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    
    # 训练循环
    best_val_loss = float('inf')
    print("开始训练...")
    
    for iter_num in tqdm(range(config.max_iters)):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        # 前向传播
        x, y = get_batch(train_data, config.batch_size, config.block_size, config.device)
        
        # 调试信息
        if iter_num == 0:
            print(f"输入形状: x={x.shape}, y={y.shape}")
            print(f"设备: {x.device}")
        
        try:
            if config.device == 'cuda':
                with ctx:
                    logits, loss = model(x, y)
            else:
                logits, loss = model(x, y)
                
            # 反向传播
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            # 记录和评估
            if iter_num % config.log_interval == 0:
                tqdm.write(f"迭代 {iter_num}: 损失 {loss.item():.4f}, 学习率 {lr:.6f}")
                
            if iter_num % config.eval_interval == 0 and iter_num > 0:
                # 简单验证
                model.eval()
                with torch.no_grad():
                    val_losses = []
                    for _ in range(config.eval_iters):
                        x_val, y_val = get_batch(train_data, config.batch_size, config.block_size, config.device)
                        if config.device == 'cuda':
                            with ctx:
                                logits_val, loss_val = model(x_val, y_val)
                        else:
                            logits_val, loss_val = model(x_val, y_val)
                        val_losses.append(loss_val.item())
                    val_loss = sum(val_losses) / len(val_losses)
                    
                tqdm.write(f"验证损失: {val_loss:.4f}")
                model.train()
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'model_best.pth')
                    print(f"保存最佳模型，验证损失: {val_loss:.4f}")
                    
            if iter_num % config.save_interval == 0 and iter_num > 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                }
                torch.save(checkpoint, f'checkpoint_{iter_num}.pth')
                print(f"保存检查点: checkpoint_{iter_num}.pth")
                
        except Exception as e:
            print(f"训练过程中出现错误 (迭代 {iter_num}): {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("训练完成!")
    torch.save(model.state_dict(), 'model_final.pth')

if __name__ == "__main__":
    main()