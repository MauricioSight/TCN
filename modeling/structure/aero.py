from torch import nn, device as TorchDevice

from logging import Logger

import torch
from modeling.structure.pytorch_base import PytorchModelStructure

class EncoderST(nn.Module):
    def __init__(self, hidden_size=64):
        super(EncoderST, self).__init__()
        self.fc1 = nn.Linear(9, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class DecoderST(nn.Module):
    def __init__(self, input_dim, hidden_size=64):
        super(DecoderST, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 9)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class EncoderP(nn.Module):
    def __init__(self, in_channels, num_levels, kernel_size=3):
        super(EncoderP, self).__init__()
        self.enc_layers = []
        for i in range(num_levels):
            in_channels = in_channels // (2 ** i)
            out_channels = in_channels // 2
            self.enc_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            if i != num_levels - 1:
                self.enc_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*self.enc_layers)

    def forward(self, x):
        x = self.encoder(x)
        return x

class DecoderP(nn.Module):
    def __init__(self, in_channels, hidden_size, input_size, num_levels, kernel_size=3):
        super(DecoderP, self).__init__()
        self.input_size = input_size
        self.first_channel = in_channels // (2 ** num_levels)

        self.fc = nn.Linear((self.first_channel * input_size) + (hidden_size * 2), self.first_channel * input_size)

        self.dec_layers = []
        for i in range(num_levels, 0, -1):
            in_ch = in_channels // (2 ** i)
            out_channels = in_channels // (2 ** (i - 1))
            self.dec_layers.append(nn.ConvTranspose1d(in_ch, out_channels, kernel_size, 
                                                      padding=kernel_size//2))
            if i != 1:
                self.dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*self.dec_layers)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.first_channel, self.input_size)
        x = self.decoder(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_size, input_size, num_levels, kernel_size=3):
        super(Encoder, self).__init__()

        self.encoder_s = EncoderST(hidden_size)
        self.encoder_p = EncoderP(in_channels, num_levels, kernel_size)
        self.encoder_t = EncoderST(hidden_size)

    def forward(self, T, P, S):
        T_emb = self.encoder_s(T) # (hidden_size,)
        P_emb = self.encoder_p(P) # 
        S_emb = self.encoder_s(S) # (hidden_size,)

        emb = torch.cat((T_emb, P_emb.flatten(start_dim=1), S_emb), dim=1)

        return emb
    
class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_size, input_size, num_levels, kernel_size=3):
        super(Decoder, self).__init__()
        self.emb_dim = hidden_size + ((in_channels // (2**num_levels)) * input_size) + hidden_size
        self.P_emb_dim = (in_channels // (2**num_levels))

        self.decoder_s = DecoderST(self.emb_dim, hidden_size)
        self.decoder_p = DecoderP(in_channels, hidden_size, input_size, num_levels, kernel_size)
        self.decoder_t = DecoderST(self.emb_dim, hidden_size)

    def forward(self, emb):
        S_t = self.decoder_s(emb)
        P_t = self.decoder_p(emb)
        T_t = self.decoder_t(emb)

        return S_t, P_t, T_t
    
class PointMapper(nn.Module):
    def __init__(self, latent_dim=704, mapped_dim=16):
        super(PointMapper, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, mapped_dim)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class AERO(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(AERO, self).__init__(config, logger, device)

        in_channels = config.get('modeling', {}).get('structure', {}).get('in_channels')
        input_size  = config.get('modeling', {}).get('structure', {}).get('input_size')
        num_levels  = config.get('modeling', {}).get('structure', {}).get('num_levels')
        hidden_size = config.get('modeling', {}).get('structure', {}).get('hidden_size')
        kernel_size = config.get('modeling', {}).get('structure', {}).get('kernel_size')
        self.in_channels = in_channels

        
        self.encoder = Encoder(in_channels, hidden_size, input_size, num_levels, kernel_size)
        self.decoder = Decoder(in_channels, hidden_size, input_size, num_levels, kernel_size)

        self.point_mapper = PointMapper(latent_dim=self.decoder.emb_dim, mapped_dim=16)


    def forward(self, x):
        batch_size = x.shape[0]

        T = x[:, :9]
        P = x[:, 9:-9]
        S = x[:, -9:]

        P = P.reshape(batch_size, self.in_channels, -1)

        emb = self.encoder(T, P, S)
        out = self.point_mapper(emb)
        
        return out

    def save_model_state_dict(self):
        torch.save(self.state_dict(), self.model_dir)

        cp_path = self.run_dir / 'criterion_point.pt'
        if hasattr(self, 'criterion_point'):
            torch.save({'criterion_point': self.criterion_point}, cp_path)


    def load_model_state_dict(self):
        if self.model_dir.exists():
            self.load_state_dict(torch.load(self.model_dir, map_location=self.device))
        
        cp_path = self.run_dir / 'criterion_point.pt'
        if cp_path.exists():
            self.criterion_point = torch.load(cp_path, map_location=self.device)['criterion_point']

    def compile(self):
        self = self.to(device=self.device)
        self.load_model_state_dict()

    def compute_criterion_point(self, data_loader):
        """
        Compute the criterion point a after point mapper pretraining.
        
        Args:
            data_loader: torch DataLoader providing data batches.
            
        Returns:
            criterion_point: torch.Tensor of shape (1, mapped_dim)
        """
        in_channels = self.config.get('modeling', {}).get('structure', {}).get('in_channels')

        self.eval()
        
        all_mapped_points = []

        with torch.no_grad():
            for _, (data, _, _) in enumerate(data_loader):
                T = data[:, :9]
                P = data[:, 9:-9]
                S = data[:, -9:]

                P = P.reshape(data.shape[0], in_channels, -1)

                # Step 1: Encode
                emb = self.encoder(T, P, S)  # H ← φ_E(S_t, θ_E)

                # Step 2: Map
                mapped_points = self.point_mapper(emb)  # M ← φ_M(H, θ_M)

                all_mapped_points.append(mapped_points)

        # Concatenate all batches
        all_mapped_points = torch.cat(all_mapped_points, dim=0)

        # Step 3: Column-wise mean (criterion point)
        criterion_point = all_mapped_points.mean(dim=0, keepdim=True)  # shape: (1, mapped_dim)

        self.criterion_point = criterion_point

        return criterion_point