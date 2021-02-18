import torch
from torch.utils.data import Dataset, DataLoader

simple_net = torch.nn.Sequential(
    torch.nn.Linear(2, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
).cuda()

#
# class RandomDataSet(Dataset):
#
#     def __init__(self, n):
#         Dataset.__init__(self)
#         self.n = n
#         self.data = torch.rand(n, 3, requires_grad=True)
#         self.labels = torch.ceil(torch.rand(n, 1))
#
#     def __len__(self):
#         return self.n
#
#     def __getitem__(self, item):
#         return self.data[item], self.labels[item]
#
#
# dl = DataLoader(RandomDataSet(1000), batch_size=100)
# opt = torch.optim.SGD(simple_net.parameters(), lr=0.001)
# for _ in range(100):
#     for batch in dl:
#         x, y = batch
#         y_pred = simple_net(x)
#         _loss = (x - y).abs().sum()
#         opt.zero_grad()
#         _loss.backward()
#         opt.step()
#
#         print(_loss)
#


from PIL import Image
import skimage
from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)
        # self.coords = torch.cat([get_mgrid(sidelength,2), torch.rand((65536, 10))], dim=-1)
        # self.coords = torch.rand((65536, 12))
        # self.coords = torch.rand(self.coords.shape)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

total_steps = 5000  # Since the whole image is our dataset, this just means 500 gradient descent steps.
optim = torch.optim.Adam(lr=1e-4, params=simple_net.parameters())

model_input, ground_truth = next(iter(dataloader))
model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

for step in range(total_steps):
    model_output = simple_net(model_input)
    loss = ((model_output - ground_truth) ** 2).mean()
    optim.zero_grad()
    loss.backward()
    optim.step()

    random_lr = torch.rand(1) / 1000
    for g in optim.param_groups:
        g['lr'] = random_lr.item()

    print(loss)
