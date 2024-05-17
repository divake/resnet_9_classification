import torch

class Config:
    def __init__(self):
        self.batch_size = 400
        self.epochs = 120
        self.max_lr = 0.001
        self.grad_clip = 0.01
        self.weight_decay = 0.001
        self.opt_func = torch.optim.Adam
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
