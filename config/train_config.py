import torch

class TrainConfig:
    # 数据配置
    data_file = "data/train_data.pt"
    batch_size = 32  # 如果GPU内存不足，可以减小这个值
    block_size = 256  # 上下文长度
    
    # 模型配置
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.1
    bias = True
    
    # 训练配置
    learning_rate = 3e-4
    max_iters = 100000
    warmup_iters = 2000
    lr_decay_iters = 100000
    min_lr = 3e-5
    
    # 设备配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    
    # 检查点
    eval_interval = 500
    eval_iters = 200
    log_interval = 10
    save_interval = 5000 
    keep_best_n_checkpoints = 3 # 只保留最好的3个检查点
    max_total_checkpoints = 5   # 最多保存5个检查点
    
    # 早停配置
    early_stop_patience = 3000  # 3000次迭代无改善则停止
    min_loss_threshold = 1.0  

    
    def __init__(self):
        # 设置新的TF32 API
        if self.device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # 使用新的API设置
            torch.backends.cuda.matmul.fp32_precision = 'tf32'
            print(f"使用设备: {self.device}, 精度: {self.dtype}")