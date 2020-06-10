import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiplicativeAttention(nn.Module):
    def __init__(self, decoder_size, encoder_size=None, attention_size = None):
        super(MultiplicativeAttention, self).__init__()
        self.decoder_size = decoder_size
        self.encoder_size = decoder_size if encoder_size == None else encoder_size
        self.attention_size = decoder_size if attention_size == None else attention_size
        self.key_map = nn.Linear(self.decoder_size, self.attention_size)
        self.value_map = nn.Linear(self.encoder_size, self.attention_size)

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        #encoder_outputs (as returned by pytorch's gru or lstm module) is of shape (seq_len, batch, hidden_size), leaving aside bidirectionality 
        #permute the encoder vectors so that batches is the 0th dimension
        encoder_outputs = encoder_outputs.permute(1,0,2)
        value = self.value_map(encoder_outputs)
        key = self.key_map(decoder_hidden)
        key = key.unsqueeze(2)
        weights = torch.bmm(value, key)
        weights = weights.squeeze(2)
        weights[~source_mask] = -float("Inf")
        # result is of shape (batch, seq_len)
        return F.softmax(weights, dim=1)

class DotproductAttention(nn.Module):
    def __init__(self):
        super(DotproductAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        #encoder_outputs (as returned by pytorch's gru or lstm module) is of shape (seq_len, batch, hidden_size), leaving aside bidirectionality 
        #permute the encoder vectors so that batches is the 0th dimension
        value = encoder_outputs.permute(1,0,2)
        key = decoder_hidden.unsqueeze(2)
        weights = torch.bmm(value, key)
        weights = weights.squeeze(2)
        weights[~source_mask] = -float("Inf")
        # result is of shape (batch, seq_len)
        return F.softmax(weights, dim=1)

class AdditiveAttention(nn.Module):
    def __init__(self, decoder_size, encoder_size=None, attention_size = None):
        super(AdditiveAttention, self).__init__()
        self.decoder_size = decoder_size
        self.encoder_size = decoder_size if encoder_size == None else encoder_size
        self.attention_size = decoder_size if attention_size == None else attention_size
        self.encoder_map = nn.Linear(self.decoder_size, self.attention_size)
        self.decoder_map = nn.Linear(self.encoder_size, self.attention_size)
        self.v = nn.Parameter(torch.FloatTensor(self.attention_size), requires_grad=True)

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        #encoder_outputs (as returned by pytorch's gru or lstm module) is of shape (seq_len, batch, hidden_size), leaving aside bidirectionality   
        #permute the encoder vectors so that batches is the 0th dimension
        encoder_outputs = encoder_outputs.permute(1,0,2)
        decoder_hidden = decoder_hidden.unsqueeze(1)
        mapped_encoder = self.encoder_map(encoder_outputs)
        mapped_decoder = self.decoder_map(decoder_hidden)
        weights = (mapped_encoder + mapped_decoder).tanh()
        weights = torch.matmul(weights, self.v)
        weights[~source_mask] = -float("Inf")      
        return F.softmax(weights, dim=1)


class NullAttention(nn.Module):
    def __init__(self):
        super(NullAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        seq_len, batch_size = encoder_outputs.shape[0], encoder_outputs.shape[1]
        weights = torch.zeros(batch_size, seq_len)
        return weights

class PositionAttention(nn.Module):
    def __init__(self, decoder_size, max_length=30):
        super(PositionAttention, self).__init__()
        self.decoder_size = decoder_size
        self.max_length = max_length
        self.attention_map = nn.Linear(self.decoder_size, self.max_length)

    def forward(self, encoder_outputs, decoder_hidden, source_mask):
        seq_len, batch_size = encoder_outputs.shape[0], encoder_outputs.shape[1]
        if seq_len > self.max_length:
            raise ValueError('input sequence too long to calculate positional attention')
        weights = self.attention_map(decoder_hidden)
        weights = weights[:, :seq_len]
        weights[~source_mask] = -float("Inf")
        return F.softmax(weights, dim=1)

def create_mask(source, vocab):
    #source is of dimensions (seq len, batch_size)
    source = source.T
    max_len = source.shape[1]
    batch_size = source.shape[0]
    pad_index = vocab.stoi['<pad>']
    mask = source.mul(source.ne(pad_index)).type(torch.bool)
    return mask
