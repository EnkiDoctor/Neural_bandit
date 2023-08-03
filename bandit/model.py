import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
       
        super(Model, self).__init__()
        self.layers = nn.ModuleList() 

        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size, output_size))
        self.layers.append(nn.ReLU())

        self.initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def initialize_weights(self):
        # 此处的初始化参数按照输入的2/dimension的方差的Normal distribution初始化
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if i < len(self.layers) - 2:  
                    #print(layer.weight.shape[1])
                    init.normal_(layer.weight, mean=0.0, std=2.0 / layer.weight.shape[1])
                else:  
                    init.normal_(layer.weight,mean = 0.0, std=1.0 / layer.weight.shape[1])
                if layer.bias is not None:
                    init.zeros_(layer.bias)