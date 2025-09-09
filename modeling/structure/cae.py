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

        # Model
        self.out_channels = 1
        self.emb_size = (hidden_size//(2**num_levels)) * (in_channels//(2**num_levels)) * (input_size//(2**num_levels))
        self.emb_shape = (hidden_size//(2**num_levels), in_channels//(2**num_levels), input_size//(2**num_levels))

        # Encoder
        self.enc_layers = []
        for i in range(num_levels):
            in_channels = 1 if i == 0 else hidden_size//(2**i)
            out_channels = hidden_size//(2**(i + 1))
            self.enc_layers.append(Encoder(in_channels=in_channels, out_channels=out_channels))

        self.encoder = nn.Sequential(*self.enc_layers)
        
        # Embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.emb_size, self.emb_size//2),
            nn.ReLU()
        )

        # Unembedding
        self.unembedding = nn.Sequential(
            nn.Linear(self.emb_size//2, self.emb_size),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=self.emb_shape)
        )

        # Decoder
        self.dec_layers = []
        for i in range(num_levels):
            in_channels = hidden_size//(2**(num_levels - i))
            out_channels = 1 if i == num_levels - 1 else hidden_size//(2**(num_levels - i - 1))

            # Calculate the output padding
            current_divisor = 2**(num_levels - i)
            next_divisor = 2**(num_levels - i - 1)
            w_size_current_size = in_channels//(current_divisor)
            w_size_next_size = in_channels//(next_divisor)
            w_size_residual = w_size_next_size - (w_size_current_size * 2)

            n_bytes_current_size = input_size//(current_divisor)
            n_bytes_next_size = input_size//(next_divisor)
            n_bytes_residual = n_bytes_next_size - (n_bytes_current_size * 2)

            input_padding  = (1 - w_size_residual, 1 - n_bytes_residual)
            output_padding = (1 - w_size_residual, 1 - n_bytes_residual)
            self.dec_layers.append(Decoder(in_channels=in_channels, out_channels=out_channels, 
                                           padding=input_padding, output_padding=output_padding))

            # If it is not the last layer, add ReLU and BatchNorm
            if i != num_levels - 1:
                self.dec_layers.append(nn.ReLU())

        self.decoder = nn.Sequential(*self.dec_layers)


    def forward(self, x):

        out = self.encoder(x.unsqueeze(1))
        out = self.embedding(out)
        out = self.unembedding(out)
        out = self.decoder(out).squeeze(1)

        return out
