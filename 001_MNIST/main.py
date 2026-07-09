import os
import torch
from torch import nn, optim, utils
import torch.nn.functional as F
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.export import export, save, load

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/MNIST")

######################
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        batch_size = x.size(0)

        assert x.size() == (batch_size, 1, 28, 28)

        out = self.conv1(x)
        assert out.size() == (batch_size, 10, 24, 24)

        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)
        assert out.size() == (batch_size, 10, 12, 12)

        out = self.conv2(out)
        assert out.size() == (batch_size, 20, 10, 10)
        out = F.relu(out)

        out = out.view(batch_size, -1)
        assert out.size() == (batch_size, 20 * 10 * 10)

        out = self.fc1(out)
        assert out.size() == (batch_size, 500)

        out = F.relu(out)
        out = self.fc2(out)
        assert out.size() == (batch_size, 10)

        # out = F.log_softmax(out, dim=1)
        # assert out.size() == (batch_size, 10)

        return out
# 捕获并可视化前20张图像
def log_images(loader, num_images=16):
    images_logged = 0
    logged_images = []
    for images, labels in loader:
        # images: batch of images, labels: batch of labels
        for i in range(images.shape[0]):
            if images_logged < num_images:
                logged_images.append(images[i])
                images_logged += 1
            else:
                break
        if images_logged >= num_images:
            break
    assert type(logged_images) == list
    print(logged_images[0].shape)
    assert logged_images[0].shape == (1, 28, 28)

    #stack make new dimension
    imgs = torch.stack(logged_images, dim = 0)

    # only onse step
    writer.add_images("MNIST Images", imgs, 0)

# train_one_epoch; epoch_index form 1
def train(model, device, train_dataloader, optimizer, criterion, epoch, num_epochs):
    model.train()

    for iter, (inputs, labels) in enumerate(train_dataloader):
        
        # to same device
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, iter + 1, len(train_dataloader),
                                                                      loss.item()))
        
        global_iter = (epoch - 1) * len(train_dataloader) + iter
        writer.add_scalar("train/loss", loss.item(), global_iter)

def test(model, device, val_dataloader, epoch):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        # 1. 循环调用val_dataloader，每次取出1个batch_size的图像和标签
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 2. 传入到resnet18模型中得到预测结果
            outputs = model(inputs)
            # 3. 获得预测的数字
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            # 4. 计算与标签一致的预测结果的数量
            correct += (predicted == labels).sum().item()
    
        # 5. 得到最终的测试准确率
        accuracy = correct / total
        # 6. 用SwanLab记录一下准确率的变化
        writer.add_scalar("val/accuracy", accuracy, global_step=epoch)


if __name__ == "__main__":


    #检测是否支持cuda
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 初始化swanlab

    config={
        "lr": 1e-4,
        "batch_size": 256,
        "num_epochs": 10,
        "device": device,
    }
    

    dataset = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
    train_dataset, val_dataset = utils.data.random_split(dataset, [55000, 5000])

    train_dataloader = utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    log_images(train_dataloader, 16)

    model = ConvNet()
    model.to(torch.device(device))


    # logsoftmax + nll
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    for epoch in range(1, config["num_epochs"]+1):
        train(model, device, train_dataloader, optimizer, criterion, epoch, config["num_epochs"])
        if epoch % 2 == 0: 
            test(model, device, val_dataloader, epoch)



    if not os.path.exists("checkpoint"):
        os.makedirs("checkpoint")
    torch.save(model.state_dict(), 'checkpoint/latest_checkpoint.pth')
