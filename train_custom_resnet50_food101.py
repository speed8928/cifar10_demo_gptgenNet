import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import Food101
from torch.utils.data import DataLoader
import os
from model.resnet import ResNet, Bottleneck

# ---------------------
# Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 24
epochs = 30
learning_rate = 0.01

# ---------------------
# Food-101 Dataset
# ---------------------
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainset = Food101(root='./data', split='train', download=True, transform=transform_train)
testset = Food101(root='./data', split='test', download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

# ---------------------
# Model: Custom ResNet-18
# ---------------------
model = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=101)
model = model.to(device)

# ---------------------
# Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ---------------------
# Training Loop
# ---------------------
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%")

# ---------------------
# Evaluation
# ---------------------
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"Test Accuracy on Food-101: {100.*correct/total:.2f}%")

# ---------------------
# Save Model
# ---------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/custom_resnet50_food101.pth")
