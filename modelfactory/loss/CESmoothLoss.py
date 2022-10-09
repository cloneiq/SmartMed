import torch
import torch.nn as nn
import torch.nn.functional as F


# 带标签平滑正则化的交叉熵损失函数
class CESmoothLoss(nn.Module):
    def __init__(self, label_smooth=None, class_num=100):
        super().__init__()
        self.label_smooth = label_smooth
        self.class_num = class_num

    def forward(self, pred, target):
        eps = 1e-12
        if self.label_smooth is not None:
            log_prob = F.log_softmax(pred, dim=1)  # softmax + log
            target = F.one_hot(target, self.class_num)  # convert to one-hot
            target = torch.clamp(target.float(), min=self.label_smooth / (self.label_smooth - 1),
                                 max=1.0 - self.label_smooth)
            loss = -1.0 * torch.sum(target * log_prob, 1)
        else:
            loss = -1.0 * pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred + eps).sum(dim=1))
        return loss.mean()


if __name__ == "__main__":
    ce_loss1 = nn.CrossEntropyLoss()
    ce_loss2 = CESmoothLoss(label_smooth=None, class_num=3)
    x = torch.tensor([[1, 8, 1], [1, 1, 8]], dtype=torch.float)
    print(x)
    y = torch.tensor([1, 2])
    print(y)
    print(ce_loss1(x, y), ce_loss2(x, y))
    ce_loss3 = CESmoothLoss(label_smooth=0.10, class_num=3)
    print(ce_loss1(x, y), ce_loss3(x, y))



