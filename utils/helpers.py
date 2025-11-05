import torch
import random

def get_batch(data, batch_size, block_size, device):
    """获取训练批次"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def generate_text(model, tokenizer, prompt, max_length=500, temperature=0.8, top_k=50):
    """生成文本"""
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_length):
            # 准备输入
            context = torch.tensor(generated[-model.config.block_size:], dtype=torch.long).unsqueeze(0)
            context = context.to(next(model.parameters()).device)
            
            # 前向传播
            logits, _ = model(context)
            logits = logits[:, -1, :] / temperature
            
            # 应用top-k过滤
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # 采样
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated.append(next_token.item())
            
            # 如果生成了结束标记，停止生成
            if next_token.item() == tokenizer.token_to_id("[SEP]"):
                break
                
    return tokenizer.decode(generated)

def estimate_loss(model, data, eval_iters, batch_size, block_size, device):
    """估计损失"""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, block_size, device)
        with torch.no_grad():
            logits, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)