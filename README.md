# cifar10_demo_gptgenNet
Improve Resnet50 by GPT4o interpreter


🔁 Benchmarking ResNets Inference on CPU

------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  23.52 M 
fwd MACs:                                                               1.2978 GMACs
fwd FLOPs:                                                              2.6073 GFLOPS
fwd+bwd MACs:                                                           3.8935 GMACs
fwd+bwd FLOPs:                                                          7.8219 GFLOPS
---------------------------------------------------------------------------------------------------
Model name: Original Model FLOPs:2.6073 GFLOPS   MACs:1.2978 GMACs   Params:23.5208 M 

Inference time for Original Model: 172.2883 seconds
Test Accuracy: 87.79%

------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  1.6 M   
fwd MACs:                                                               101.551 MMACs
fwd FLOPs:                                                              203.777 MFLOPS
fwd+bwd MACs:                                                           304.652 MMACs
fwd+bwd FLOPs:                                                          611.331 MFLOPS
---------------------------------------------------------------------------------------------------
Model name: Lite Model FLOPs:203.777 MFLOPS   MACs:101.551 MMACs   Params:1.5984 M 

Inference time for Lite Model: 12.4326 seconds
Test Accuracy: 90.24%

------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  7.68 K  
fwd MACs:                                                               0 MACs  
fwd FLOPs:                                                              524.288 KFLOPS
fwd+bwd MACs:                                                           0 MACs  
fwd+bwd FLOPs:                                                          1.5729 MFLOPS
---------------------------------------------------------------------------------------------------
Model name: Quant Model FLOPs:524.288 KFLOPS   MACs:0 MACs   Params:7.68 K 

Inference time for Quant Model: 10.7362 seconds
Test Accuracy: 79.81%

------------------------------------- Calculate Flops Results -------------------------------------
Notations:
number of parameters (Params), number of multiply-accumulate operations(MACs),
number of floating-point operations (FLOPs), floating-point operations per second (FLOPS),
fwd FLOPs (model forward propagation FLOPs), bwd FLOPs (model backward propagation FLOPs),
default model backpropagation takes 2.00 times as much computation as forward propagation.

Total Training Params:                                                  896     
fwd MACs:                                                               0 MACs  
fwd FLOPs:                                                              90.112 KFLOPS
fwd+bwd MACs:                                                           0 MACs  
fwd+bwd FLOPs:                                                          270.336 KFLOPS
---------------------------------------------------------------------------------------------------
Model name: Quant Lite Model FLOPs:90.112 KFLOPS   MACs:0 MACs   Params:896 

Inference time for Quant Lite Model: 5.0032 seconds
Test Accuracy: 82.57%

