import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from torchvision.datasets import FashionMNIST
from torch.optim.lr_scheduler import CosineAnnealingLR

from resnet import ResNet


input_size = 28 * 28
num_classes = 10

seed = 3407
num_epochs = 300
batch_size = 2048
learning_rate = 5e-3
weight_decay = 2e-4
val_interval = 5
load_pth = None
save_pth = 'model.pth'

dataset_train = FashionMNIST(root='./data',
                             train=True,
                             transform=v2.Compose([
                                 v2.ToImage(),
                                 v2.RandomHorizontalFlip(),
                                 v2.Resize((16, 16)),
                                 v2.ToDtype(torch.float32, scale=True),
                             ]),
                             download=True)
dataset_test = FashionMNIST(root='./data',
                            train=False,
                            transform=v2.Compose([
                                v2.ToImage(),
                                v2.Resize((16, 16)),
                                v2.ToDtype(torch.float32, scale=True),
                            ]))
dataloader_train = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              shuffle=True)
dataloader_test = DataLoader(dataset=dataset_test,
                             batch_size=4096,
                             shuffle=False)


model = ResNet(10).cuda()
if load_pth is not None:
    model.load_state_dict(torch.load(load_pth))
for param in model.parameters():
    param = param.to(torch.float16)
criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, weight_decay=weight_decay)
scheduler = CosineAnnealingLR(optimizer, num_epochs)


def single_train(train_model, train_criterion, train_optimizer, train_dataloader):
    train_model.train()
    cum_loss, cum_loss_idx, loss = 0., 0, None
    bar = tqdm(train_dataloader)
    bar.set_description('Train')
    for x, y in bar:
        x, y = x.cuda(), y.cuda()
        pred = train_model(x)
        loss = train_criterion(pred, y)
        cum_loss += loss.item()
        cum_loss_idx += 1

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
    bar.close()
    avg_loss = cum_loss / cum_loss_idx
    return avg_loss


@torch.no_grad()
def single_test(test_model, test_criterion, test_dataloader):
    test_model.eval()
    cum_loss, cum_loss_idx, cum_correct_num, cum_total_num = 0., 0, 0, 0
    bar = tqdm(test_dataloader)
    bar.set_description('Test')
    for x, y in bar:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = test_criterion(pred, y)
        pred_argmax = torch.argmax(pred, dim=-1)
        cum_loss += loss.item()
        cum_loss_idx += 1
        cum_correct_num += torch.sum(pred_argmax == y).item()
        cum_total_num += y.shape[0]
    bar.close()
    avg_loss, avg_acc = cum_loss / cum_loss_idx, cum_correct_num / cum_total_num
    return avg_loss, avg_acc


def main():
    torch.manual_seed(seed)
    model_param = sum(param.numel() for param in model.parameters())
    best_acc = 0.
    print('model_param:', model_param)
    for epoch_idx in range(num_epochs):
        train_loss = single_train(train_model=model,
                                  train_criterion=criterion,
                                  train_optimizer=optimizer,
                                  train_dataloader=dataloader_train)
        print("\r\033[K", end="")  # remove the residual bar and information
        print("\033[A", end="")  # return to last line
        print("\r\033[K", end="")
        if (epoch_idx + 1) % val_interval == 0:
            test_loss, test_acc = single_test(test_model=model,
                                              test_criterion=criterion,
                                              test_dataloader=dataloader_test)
            print("\r\033[K", end="")  # remove the residual bar and information
            print("\033[A", end="")  # return to last line
            print("\r\033[K", end="")
            
            if test_acc > best_acc:
                torch.save(model.state_dict(), save_pth)
                best_acc = test_acc
        print(f'epoch ', epoch_idx + 1, 'train_loss', train_loss)
        if (epoch_idx + 1) % val_interval == 0:
            print(f'epoch ', epoch_idx + 1, 'test_loss', test_loss, 'test_acc', test_acc)
        scheduler.step()
    


if __name__ == '__main__':
    main()
