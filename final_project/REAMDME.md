# CS-230 Final Project

best_model_weights folder

Install dep `pip install -r requirements.txt`



final_project/
├── b/
├── src/
│   └── data_synthesis.py
└── README.md


### Known limitations
 - drop some feature that could be useful, merely working on transaction description 
 -  


 Model Architecture:
TransactionClassifier(
  (layers): Sequential(
    (0): Linear(in_features=1003, out_features=256, bias=True)
    (1): ReLU()
    (2): BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.3, inplace=False)
    (4): Linear(in_features=256, out_features=128, bias=True)
    (5): ReLU()
    (6): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.2, inplace=False)
    (8): Linear(in_features=128, out_features=64, bias=True)
    (9): ReLU()
    (10): BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (11): Dropout(p=0.1, inplace=False)
    (12): Linear(in_features=64, out_features=32, bias=True)
    (13): ReLU()
    (14): BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (15): Dropout(p=0.1, inplace=False)
    (16): Linear(in_features=32, out_features=24, bias=True)
  )
)

Total Parameters: 302,008
Trainable Parameters: 302,008

Detailed Layer Information:

Layer: layers.0.weight
Shape: torch.Size([256, 1003])
Parameters: 256,768
Mean: -0.000011
Std: 0.018239
Min: -0.031575
Max: 0.031575

Layer: layers.0.bias
Shape: torch.Size([256])
Parameters: 256
Mean: -0.001016
Std: 0.017363
Min: -0.031091
Max: 0.031481

Layer: layers.2.weight
Shape: torch.Size([256])
Parameters: 256
Mean: 1.000000
Std: 0.000000
Min: 1.000000
Max: 1.000000

Layer: layers.2.bias
Shape: torch.Size([256])
Parameters: 256
Mean: 0.000000
Std: 0.000000
Min: 0.000000
Max: 0.000000

Layer: layers.4.weight
Shape: torch.Size([128, 256])
Parameters: 32,768
Mean: -0.000434
Std: 0.036053
Min: -0.062498
Max: 0.062496

Layer: layers.4.bias
Shape: torch.Size([128])
Parameters: 128
Mean: 0.001317
Std: 0.039120
Min: -0.062387
Max: 0.061752

Layer: layers.6.weight
Shape: torch.Size([128])
Parameters: 128
Mean: 1.000000
Std: 0.000000
Min: 1.000000
Max: 1.000000

Layer: layers.6.bias
Shape: torch.Size([128])
Parameters: 128
Mean: 0.000000
Std: 0.000000
Min: 0.000000
Max: 0.000000

Layer: layers.8.weight
Shape: torch.Size([64, 128])
Parameters: 8,192
Mean: 0.000178
Std: 0.050570
Min: -0.088386
Max: 0.088385

Layer: layers.8.bias
Shape: torch.Size([64])
Parameters: 64
Mean: 0.002561
Std: 0.052245
Min: -0.083869
Max: 0.086300

Layer: layers.10.weight
Shape: torch.Size([64])
Parameters: 64
Mean: 1.000000
Std: 0.000000
Min: 1.000000
Max: 1.000000

Layer: layers.10.bias
Shape: torch.Size([64])
Parameters: 64
Mean: 0.000000
Std: 0.000000
Min: 0.000000
Max: 0.000000

Layer: layers.12.weight
Shape: torch.Size([32, 64])
Parameters: 2,048
Mean: 0.001203
Std: 0.070755
Min: -0.124927
Max: 0.124874

Layer: layers.12.bias
Shape: torch.Size([32])
Parameters: 32
Mean: 0.012784
Std: 0.071248
Min: -0.110342
Max: 0.124176

Layer: layers.14.weight
Shape: torch.Size([32])
Parameters: 32
Mean: 1.000000
Std: 0.000000
Min: 1.000000
Max: 1.000000

Layer: layers.14.bias
Shape: torch.Size([32])
Parameters: 32
Mean: 0.000000
Std: 0.000000
Min: 0.000000
Max: 0.000000

Layer: layers.16.weight
Shape: torch.Size([24, 32])
Parameters: 768
Mean: 0.003704
Std: 0.104204
Min: -0.176667
Max: 0.175409

Layer: layers.16.bias
Shape: torch.Size([24])
Parameters: 24
Mean: -0.013933
Std: 0.111068
Min: -0.173551
Max: 0.171797