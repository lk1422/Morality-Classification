import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PosEnc(nn.Module):
    def __init__(self, d_m, dropout=0.2, size_limit=5000):
        super(PosEnc, self).__init__()
        self.dropout = nn.Dropout(dropout)
        p_enc = torch.zeros(size_limit, d_m)
        pos = torch.arange(0, size_limit, dtype=torch.float).unsqueeze(1)
        divider = torch.exp(torch.arange(0, d_m, 2).float() * (-math.log(10000.0) / d_m))
        # divider is the list of radians, multiplied by position indices of words, and fed to the sinusoidal and cosinusoidal function
        p_enc[:, 0::2] = torch.sin(pos * divider)
        p_enc[:, 1::2] = torch.cos(pos * divider)
        p_enc = p_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('p_enc', p_enc)

    def forward(self, x):
        return self.dropout(x + self.p_enc[:x.size(0), :])

class Transformer(nn.Module):
    def __init__(self, num_tokens, emb_dim, max_seq, num_heads, num_hidden, num_layers, dropout=0.3, device=None):
        super(Transformer,self).__init__()

        self.enc = nn.Embedding(num_tokens, emb_dim)
        self.pos_enc = nn.Embedding(max_seq, emb_dim)

        layers = TransformerEncoderLayer(emb_dim, num_heads, num_hidden, dropout, batch_first=True)
        self.trans = TransformerEncoder(layers, num_layers)
        dec_lay = nn.TransformerDecoderLayer(emb_dim, num_heads, batch_first=True)
        self.dec = nn.TransformerDecoder(dec_lay, num_layers)

        self.dec1 = nn.Linear(emb_dim*max_seq, 1)

        self.emb = emb_dim
        self.time_steps = torch.LongTensor([[i for i in range(max_seq)]])
        self.time_steps = self.time_steps.to(device)

        self.dropout = nn.Dropout((dropout))

    def forward(self, source):
        pos = self.pos_enc(self.time_steps)
        pos.expand(source.shape[0], -1, -1)
        input_tensor = self.enc(source) * math.sqrt(self.emb) + pos
        input_tensor = self.dropout(input_tensor)

        out = self.trans(input_tensor)
        out = self.dec(input_tensor, out)
        out = out.reshape(out.shape[0], -1)
        out = self.dec1(out)

        return out


class Transformer2(nn.Module):
    def __init__(self, max_len, num_token, emb_dim, num_heads, num_hidden, num_layers, dropout=0.3):
        super(Transformer2, self).__init__()
        self.model_name = 'transformer'
        self.position_enc = PosEnc(emb_dim, dropout, max_len)
        layers_enc = TransformerEncoderLayer(emb_dim, num_heads, num_hidden, dropout)
        self.enc_transformer = TransformerEncoder(layers_enc, num_layers)
        self.enc = nn.Embedding(num_token, emb_dim)
        self.emb_dim = emb_dim
        self.dec = nn.Linear(max_len*emb_dim, 1)
        self.init_params()

    def init_params(self):
        initial_rng = 0.12
        self.enc.weight.data.uniform_(-initial_rng, initial_rng)
        self.dec.bias.data.zero_()
        self.dec.weight.data.uniform_(-initial_rng, initial_rng)

    def forward(self, source):
        source = self.enc(source) * math.sqrt(self.emb_dim)
        source = self.position_enc(source)
        op = self.enc_transformer(source)
        op = op.reshape(op.shape[0], -1)
        op = torch.sigmoid(self.dec(op))
        return op

class SimpleModel1(nn.Module):
    def __init__(self, max_len, emb_dim, num_tokens):
        super(SimpleModel1, self).__init__()
        self.dropout = nn.Dropout()
        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.l1 = nn.Linear(max_len*emb_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1)
    def forward(self, x):
        x = self.dropout(self.emb(x))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

class SimpleModel2(nn.Module):
    def __init__(self, max_len, emb_dim, num_tokens):
        super(SimpleModel2, self).__init__()
        self.dropout = nn.Dropout()
        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.l1 = nn.Linear(max_len*emb_dim, 1024)
        self.l2 = nn.Linear(1024, 1024)
        self.l3 = nn.Linear(1024, 1)
    def forward(self, x):
        x = self.emb(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.l1(x))
        x = self.dropout(F.relu(self.l2(x)) + x)
        x = torch.sigmoid(self.l3(x))
        return x

class SimpleModel(nn.Module):
    def __init__(self, max_len, emb_dim, num_tokens):
        super(SimpleModel, self).__init__()
        self.dropout = nn.Dropout()
        self.emb = nn.Embedding(num_tokens, emb_dim)
        self.l1 = nn.Linear(max_len*emb_dim, max_len*emb_dim//2)
        self.l2 = nn.Linear(max_len*emb_dim//2, 1)
    def forward(self, x):
        x = self.emb(x)
        x = self.dropout(x.reshape(x.shape[0], -1))
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

if __name__ == "__main__":
    model = Transformer2(max_len=200 ,num_tokens=len(dictionary), emb_dim=256, num_heads=8, num_hidden=2, num_layers=3, device=device)
    

        
        
