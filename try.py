import torch
from torch.utils.data import Dataset, DataLoader

simple_net = torch.nn.Sequential(
    torch.nn.Linear(3, 128),
    torch.nn.ReLU(),
    # torch.nn.Linear(256, 256),
    # torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)


class RandomDataSet(Dataset):

    def __init__(self, n):
        Dataset.__init__(self)
        self.n = n
        self.data = torch.rand(n, 3, requires_grad=True)
        self.labels = torch.ceil(torch.rand(n, 1))

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


dl = DataLoader(RandomDataSet(1000), batch_size=100)
opt = torch.optim.SGD(simple_net.parameters(), lr=0.001)
for _ in range(100):
    for batch in dl:
        x, y = batch
        y_pred = simple_net(x)
        _loss = (x - y).abs().sum()
        opt.zero_grad()
        _loss.backward()
        opt.step()

        print(_loss)
