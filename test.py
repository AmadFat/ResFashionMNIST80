import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import FashionMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

from resnet import ResNet


@torch.no_grad()
def single_test(test_model, test_criterion, test_dataloader):
    test_model.eval()
    cum_loss, cum_loss_idx, cum_correct_num, cum_total_num = 0., 0, 0, 0
    bar = tqdm(test_dataloader)
    bar.set_description('Test')
    for x, y in bar:
        x, y = x.cuda(), y.cuda()
        pred = test_model(x)
        loss = test_criterion(pred, y)
        pred_argmax = torch.argmax(pred, dim=-1)
        cum_loss += loss.item()
        cum_loss_idx += 1
        cum_correct_num += torch.sum(pred_argmax == y).item()
        cum_total_num += y.shape[0]
    bar.close()
    avg_loss, avg_acc = cum_loss / cum_loss_idx, cum_correct_num / cum_total_num
    return avg_loss, avg_acc


model = ResNet(10).cuda()
model.load_state_dict(torch.load('model.pth'))
criterion = nn.CrossEntropyLoss()
dataset_test = FashionMNIST(root='./data',
                            train=False,
                            transform=v2.Compose([
                                v2.ToImage(),
                                v2.Resize((16, 16)),
                                v2.ToDtype(torch.float32, scale=True),
                            ]),
                            download=True)
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=4096,
                             shuffle=False)


def main():
    _, test_acc = single_test(test_model=model,
                              test_criterion=criterion,
                              test_dataloader=dataloader_test)
    print("\r\033[K", end="")  # remove the residual bar and information
    print("\033[A", end="")  # return to last line
    print("\r\033[K", end="")
    print('test_acc', test_acc)


if __name__ == '__main__':
    main()
