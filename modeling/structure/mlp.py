from torch import nn, device as TorchDevice

from logging import Logger
from modeling.structure.pytorch_base import PytorchModelStructure

class MLP(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(MLP, self).__init__(config, logger, device)

        input_size  = config.get('modeling', {}).get('structure', {}).get('input_size')
        num_layers  = config.get('modeling', {}).get('structure', {}).get('num_layers')
        hidden_size = config.get('modeling', {}).get('structure', {}).get('hidden_size')
        output_size = config.get('modeling', {}).get('structure', {}).get('output_size')

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = []

        layers = []
        for i in range(num_layers):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        self.hidden_layers = nn.Sequential(*layers)

        self.output_layer = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)

        out = self.input_layer(x)
        out = self.hidden_layers(out)
        out = self.output_layer(out)
        return out
