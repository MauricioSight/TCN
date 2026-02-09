from torch import nn, device as TorchDevice

from logging import Logger
from modeling.structure.pytorch_base import PytorchModelStructure

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding=(1, 1)):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, padding=(1, 1), output_padding=(1, 1)):
        super(Decoder, self).__init__()
        self.decoder = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, 
                                                      padding=padding, output_padding=output_padding)

    def forward(self, x):
        x = self.decoder(x)
        return x

class CAE(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(CAE, self).__init__(config, logger, device)

        in_channels = config.get('modeling', {}).get('structure', {}).get('in_channels')
        input_size  = config.get('modeling', {}).get('structure', {}).get('input_size')
        num_levels  = config.get('modeling', {}).get('structure', {}).get('num_levels')
        hidden_size = config.get('modeling', {}).get('structure', {}).get('hidden_size')

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, (hidden_size//4), kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d((hidden_size//4), eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d((hidden_size//4), (hidden_size//2), kernel_size=3, stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d((hidden_size//2), eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d((hidden_size//2), hidden_size, kernel_size=3, stride=1, padding=(1, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size, eps=0.001, momentum=0.9),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_size * in_channels, hidden_size * in_channels//2), # CHANGE TO THIS WHEN RUNNING SAD: nn.Linear(hidden_size * 4, hidden_size * 4//2), # 
            nn.ReLU()
        )

        # Unembedding
        self.unembedding = nn.Sequential(
            nn.Linear(hidden_size * in_channels//2, hidden_size * in_channels), # CHANGE TO THIS WHEN RUNNING SAD: nn.Linear(hidden_size*4//2, hidden_size*4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(hidden_size, in_channels // 8, 8)), # CHANGE TO THIS WHEN RUNNING SAD: nn.Unflatten(dim=1, unflattened_size=(hidden_size, 2, 2))
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_size, out_channels=(hidden_size//2), kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(hidden_size//2), out_channels=(hidden_size//4), kernel_size=3, stride=2, padding=1, output_padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=(hidden_size//4), out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=(1, 1)), # CHANGE TO THIS WHEN RUNNING SAD: nn.ConvTranspose2d(in_channels=(hidden_size//4), out_channels=1, kernel_size=3, stride=2, padding=(1, 2), output_padding=(1, 1)),
        )


    def forward(self, x):

        out = self.encoder(x.unsqueeze(1))
        out = self.embedding(out)
        out = self.unembedding(out)
        out = self.decoder(out).squeeze(1)

        return out