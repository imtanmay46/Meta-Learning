import os
import json
import random
import torch
import numpy as np
import torch.nn as nn
import mplcyberpunk as mcy
from tqdm.auto import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap, Normalize

class EncoderDecoderModel(nn.Module):
    def __init__(self, input_size=900, hidden_size=512, intermediate_size=256, encoded_size=128, num_heads=4, dropout_rate=0.3):
        super().__init__()

        self.encoder_fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(intermediate_size, encoded_size),
            nn.BatchNorm1d(encoded_size),
            nn.ReLU(),
        )
        
        self.encoder_attention = nn.MultiheadAttention(embed_dim=encoded_size, num_heads=num_heads, dropout=dropout_rate, batch_first=True)

        self.decoder_attention = nn.MultiheadAttention(embed_dim=encoded_size, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(encoded_size, intermediate_size),
            nn.BatchNorm1d(intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(intermediate_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), 1, -1)
        encoded_fc = self.encoder_fc(x.squeeze(1))
        encoded_fc = encoded_fc.unsqueeze(1)
        
        encoded, _ = self.encoder_attention(encoded_fc, encoded_fc, encoded_fc)
        decoded_attention, _ = self.decoder_attention(encoded, encoded, encoded)
        
        decoded_attention = decoded_attention.squeeze(1)
        decoded = self.decoder_fc(decoded_attention)
        
        return decoded