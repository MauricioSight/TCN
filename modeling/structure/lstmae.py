from torch import nn, device as TorchDevice

from logging import Logger
from modeling.structure.pytorch_base import PytorchModelStructure


class LSTMAE(PytorchModelStructure):
    def __init__(self, config: dict, logger: Logger, device: TorchDevice):
        super(LSTMAE, self).__init__(config, logger, device)

        input_size  = config.get('modeling', {}).get('structure', {}).get('input_size')
        hidden_size = config.get('modeling', {}).get('structure', {}).get('hidden_size')

        # Encoder
        self.encoder_lstm1 = nn.LSTM(input_size=input_size, hidden_size=(hidden_size*2), batch_first=True)
        self.encoder_lstm2 = nn.LSTM(input_size=(hidden_size*2), hidden_size=hidden_size, batch_first=True)

        # Decoder
        self.decoder_lstm1 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.decoder_lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=(hidden_size*2), batch_first=True)
        self.decoder_linear = nn.Linear((hidden_size*2), input_size)


    def forward(self, x):
        # (N, seq, n_bytes)
        w = x.shape[1]

        # Encoder
        out, (hn, cn) = self.encoder_lstm1(x)
        out, (hn, cn) = self.encoder_lstm2(out)
        out = hn  # Use only the last hidden state for the embedding vector

        # Repeat the embedding vector w times
        out = out.repeat(w, 1, 1).permute(1, 0, 2)

        # Decoder
        out, _ = self.decoder_lstm1(out)
        out, _ = self.decoder_lstm2(out)
        out = self.decoder_linear(out)

        return out
