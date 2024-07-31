import torch
import torch.nn as nn


torch.cuda.set_device(0)


def _mean_update(pre_mean, pre_count, x, pre_cap=None):
    this_sum = torch.sum(x, 0)
    this_batch_size = float(x.shape[0])

    new_count = pre_count + this_batch_size
    alpha = this_batch_size / torch.minimum(new_count, pre_cap)

    new_mean = pre_mean * (1 - alpha) + (this_sum / this_batch_size) * alpha

    return new_mean, new_count


class MeanStream(nn.Module):
    def __init__(self, input_shape=(1, 3, 80, 96, 80), cap=100):
        super(MeanStream, self).__init__()
        self.cap = torch.tensor(float(cap))
        self.mean = torch.zeros(input_shape[1:]).cuda()
        self.mean.requires_grad = False
        self.count = torch.tensor(0.0).cuda()
        self.count.requires_grad = False
        self.input_shape = input_shape

    def forward(self, x):
        device = x.device
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        self.mean.data = new_mean.data
        self.count.data = new_count.data

        p = self.input_shape
        z = torch.ones(p).to(device)

        return torch.minimum(torch.tensor(1.), new_count / self.cap) * (z * new_mean.unsqueeze(dim=0))
