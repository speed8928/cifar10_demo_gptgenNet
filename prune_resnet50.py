import torch
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import os
from model.resnet import ResNet, Bottleneck

# ---------------------
# Load Trained ResNet-50
# ---------------------
device = torch.device("cpu")
model = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=10)
model.load_state_dict(torch.load("checkpoints/resnet50_cifar10.pth", map_location=device))
model.eval()

# ---------------------
# Apply Global Unstructured Pruning (e.g., 30%)
# ---------------------
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
        parameters_to_prune.append((module, "weight"))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
print(parameters_to_prune)
# Remove pruning re-parametrizations (make pruning permanent)
for module, _ in parameters_to_prune:
    prune.remove(module, "weight")

# ---------------------
# Evaluate Pruned Model
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, targets in testloader:
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"Pruned Accuracy: {100. * correct / total:.2f}%")

# ---------------------
# Save Pruned Model
# ---------------------
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "checkpoints/resnet50_cifar10_pruned.pth")
