import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim

#hyperparameters
epochs = 20
momentum = 0.5
batch_size = 64
learning_rate = 0.01


def train(model, loader, optimizer, epoch):
    model.train()
    running_loss = 0
    for (data, target) in loader:
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        #gradient of loss
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print(f'Training Epoch {epoch}: loss={running_loss/len(loader.dataset)}')

def test(model, loader):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for (data, target) in loader:
            output = model(data)
            running_loss += F.nll_loss(output, target, reduction='sum').item()
            predicted = output.argmax(dim=1)
            correct += predicted.eq(target).sum().item()

    length = len(loader.dataset)
    loss = running_loss / length
    
    print(f'Test: Average loss = {loss}, Accuracy = {correct}/{length} ({100.*correct/length}%)')


if __name__ == "__main__":
    transform = transforms.ToTensor()
    
    train_set = datasets.FashionMNIST('dataset', train=True, download=True,
                           transform=transform)
    test_set = datasets.FashionMNIST('dataset', train=False,
                           transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)
    model = nn.Sequential(
        #First conv. layer + ReLU
        nn.Conv2d(in_channels=1, out_channels=16,kernel_size=3,stride=1),
        nn.ReLU(),
        
        #Second conv. layer + ReLU
        nn.Conv2d(in_channels=16, out_channels=32,kernel_size=3,stride=1),
        nn.ReLU(),
        
        #pool layer
        nn.MaxPool2d(kernel_size = 2),
        
        #Flatten
        nn.Flatten(),
        
        #Fully connected layer    
        nn.Linear(4608, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
        )

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    last_epoch = 0
    try:
        checkpoint = torch.load("checkpoint.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        print("success")
    except:
        pass

    for i in range(last_epoch+1, epochs+1):
        train(model, train_loader, optimizer, i)
        test(model, test_loader)

    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        }, "checkpoint.pt")

    torch.save(model,"model.pt")