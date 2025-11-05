class GPTConfig:
    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = kwargs.get('n_layer', 12)
        self.n_head = kwargs.get('n_head', 12)
        self.n_embd = kwargs.get('n_embd', 768)
        self.dropout = kwargs.get('dropout', 0.1)
        self.bias = kwargs.get('bias', True)