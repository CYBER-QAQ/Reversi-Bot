from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('tb-logs')
for i in torch.arange(10):
    mean = torch.sin(i/2.0)
    random_weights = torch.normal(mean, 0.1, (100,))
    writer.add_histogram('Weight', random_weights, i)
    # writer.add_scalars('Loss', {'x': torch.sin(i / 5.0), 'y': torch.cos(i / 5.0)}, i)

writer.close()
