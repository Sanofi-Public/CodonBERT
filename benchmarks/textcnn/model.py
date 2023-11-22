import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):
    def __init__(self, args):
        super(CNN_Text, self).__init__()

        D = args.get("embed_dim")
        C = args.get("labels")
        Ci = 1
        Co = args.get("kernel_num")
        Ks = args.get("kernel_sizes")

        self.pre_embed = False
        if args.get("embed_file"):
            self.pre_embed = True
        else:
            V = args.get("vocab_size")
            self.embed = nn.Embedding(V, D)

            if args.get("static"):
                self.embed.weight.requires_grad = False

        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        self.dropout = nn.Dropout(args.get("dropout"))
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def forward(self, x):
        if not self.pre_embed:
            x = self.embed(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [
            F.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # [(N, Co, W), ...]*len(Ks)
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)

        return logit
