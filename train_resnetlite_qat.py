import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import torch.ao.quantization as quantization
from model.resnetlite_quant import ResNetLiteQuant, BasicBlockLite

# ---------------------
# Setup
# ---------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 30
learning_rate = 0.01

# ---------------------
# CIFAR-10 Dataset
# ---------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# ---------------------
# Model
# ---------------------
model = ResNetLiteQuant(num_classes=10)
model = model.to(device)

# ---------------------
# Fuse modules before QAT
# ---------------------
model.eval()
model.fuse_model()

# ---------------------
# Prepare for QAT
# ---------------------
model.train()
torch.backends.quantized.engine = "fbgemm"
model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
quantization.prepare_qat(model, inplace=True)

# ---------------------
# Loss & Optimizer
# ---------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---------------------
# QAT Training Loop
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
# Convert to Quantized Model
# ---------------------
model.eval()
quantized_model = quantization.convert(model.cpu(), inplace=True)

# ---------------------
# Evaluate
# ---------------------
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to('cpu'), targets.to('cpu')
        outputs = quantized_model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"✅ Quantized Test Accuracy: {100.*correct/total:.2f}%")

# ---------------------
# Save Model
# ---------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(quantized_model.state_dict(), "checkpoints/resnetlite_cifar10_qat_final.pth")
