import torch
import torch.nn as nn
from dataset.asr_vocab import VOCAB_SIZE


class ASRModel(nn.Module):
    
    def __init__(self, n_mels: int = 80, hidden_size: int = 128):
        super().__init__()

        self.encoder = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            VOCAB_SIZE
        )
        
        with torch.no_grad():
            self.classifier.bias[0] = -2.0        

    def forward(self, x):
        hidden_states, _ = self.encoder(x)
        logits = self.classifier(hidden_states)
        return logits
    
    
    

        
        
        