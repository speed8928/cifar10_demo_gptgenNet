import torch
import time
import torchvision
import torchvision.transforms as transforms
from model.resnet import ResNet, Bottleneck
from model.resnet_lite import ResNetLite
from calflops import calculate_flops
# ---------------------
# Load CIFAR-10 test data
# ---------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1024, shuffle=False)

device = torch.device("cpu")

# ---------------------
# Load original model
# ---------------------
model_orig = ResNet(block=Bottleneck, num_blocks=[3, 4, 6, 3], num_classes=10)
model_orig.load_state_dict(torch.load("checkpoints/resnet50_cifar10.pth", map_location=device))
model_orig.eval()

# ---------------------
# Load pruned model
# ---------------------

model_pruned = ResNetLite(num_classes=10)
model_pruned.load_state_dict(torch.load("checkpoints/resnet_lite_cifar10.pth", map_location=device))
model_pruned.eval()

# ---------------------
# Function to time inference
# ---------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def benchmark(model, name):


    input_shape = (1, 3, 32, 32)
    flops, macs, params = calculate_flops(model=model, 
                                      input_shape=input_shape,
                                      output_as_string=True,
                                      output_precision=4,
                                      print_detailed=False)
               
    print("Model name: %s FLOPs:%s   MACs:%s   Params:%s \n" %(name, flops, macs, params))


    paras = count_parameters(model)
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for inputs, targets in testloader:
            outputs = model(inputs)
          
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    end = time.time()
    #print(f"Model parameters for {name}: {paras} ")
    print(f"Inference time for {name}: {end - start:.4f} seconds")
    print(f"Test Accuracy: {100.*correct/total:.2f}%")
# ---------------------
# Run Benchmark
# ---------------------
print("🔁 Benchmarking ResNet-50 Inference on CPU")
benchmark(model_orig, "Original Model")
benchmark(model_pruned, "Lite Model")
