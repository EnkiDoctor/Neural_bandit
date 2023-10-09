import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
       
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers


        self.layers = nn.ModuleList() 
        self.layers.append(nn.Linear(input_size*2, hidden_size*2))
        self.layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size*2, hidden_size*2))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(hidden_size*2, output_size))
        self.layers.append(nn.ReLU())

        self.initialize_weights()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # x multiply by output_size
        x = x * self.output_size
        return x
    
    def initialize_weights(self):
        def initial_weights(dim):
            w = []
            total_dim = 0
            for i in range(0, len(dim) - 1):
                if i < len(dim) - 2:
                    temp = torch.randn(dim[i + 1], dim[i]) / torch.sqrt(dim[i + 1])
                    temp = torch.kron(torch.eye(2, dtype=torch.int), temp)
                    w.append(temp)
                    total_dim += dim[i + 1] * dim[i] * 4
                else:
                    temp = torch.randn(dim[i + 1], dim[i]) / torch.sqrt(dim[i])
                    temp = torch.kron(torch.tensor([[1, -1]], dtype=torch.float), temp)
                    w.append(temp)
                    total_dim += dim[i + 1] * dim[i] * 2

            return w, total_dim
        
        def INI(dim):  
            #### initialization
            #### dim consists of (d1, d2,...), where dl = 1 (placeholder, deprecated)
            w = []
            total_dim = 0
            for i in range(0, len(dim) - 1):
                if i < len(dim) - 2:
                    temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i + 1])
                    temp = np.kron(np.eye(2, dtype=int), temp)
                    temp = torch.from_numpy(temp).to(torch.float32)
                    w.append(temp)
                    total_dim += dim[i + 1] * dim[i] *4
                else:
                    temp = np.random.randn(dim[i + 1], dim[i]) / np.sqrt(dim[i])
                    temp = np.kron([[1, -1]], temp)
                    temp = torch.from_numpy(temp).to(torch.float32)
                    w.append(temp)
                    total_dim += dim[i + 1] * dim[i]*2

            return w, total_dim




        input_size = self.input_size
        hidden_sizes = [self.hidden_size for layer in range(self.num_layers)]
        output_size = self.output_size

        dim_tensor = torch.tensor([input_size] + hidden_sizes + [output_size], dtype=torch.int)  # 将列表转换为Tensor

        print(dim_tensor)
       # w = initial_weights(dim_tensor)
        w,total_dim = INI(dim_tensor)
    
        idx = 0
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                layer.weight.data = w[idx]
                idx += 1
