import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Seq2Seq(nn.Module):
    def __init__(self, input_text, embedding_size, hidden_size, output_text,
                 attention='Null'):
        super(Seq2Seq, self).__init__()
        self.input_vocab = input_text.vocab
        self.output_vocab = output_text.vocab
        self.encoder = Encoder(len(self.input_vocab), embedding_size,
                               hidden_size)
        self.decoder = Decoder(self.output_vocab, embedding_size, hidden_size,
                               self.input_vocab, attention=attention)
        self.attention_string = attention

    def forward(self, source, target, teacher_force=0.5):
        output, final_hidden = self.encoder(source)
        decoder_outputs, decoder_hiddens, _ = self.decoder(target,
                                                           final_hidden,
                                                           output, source,
                                                           teacher_force=teacher_force)
        return decoder_outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size)

    def forward(self, input):
        batch_size = input.shape[0]
        embedded = self.embedding(input)
        output, final_hidden = self.rnn(embedded)
        return output, final_hidden


class Decoder(nn.Module):
    def __init__(self, vocab, embedding_size, hidden_size, input_vocab,
                 attention):
        super(Decoder, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.input_vocab = input_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.grucell = nn.GRUCell(
            embedding_size + (hidden_size if attention else 0), hidden_size)
        self.embedding_out = nn.Linear(hidden_size, self.vocab_size)
        if attention == 'Null':
            self.attention = NullAttention()
        elif attention == 'Position':
            self.attention = PositionAttention(hidden_size)
        elif attention == 'Additive':
            self.attention = AdditiveAttention(hidden_size)
        elif attention == 'Multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)

    def forward(self, target, init_hidden, encoder_hiddens, source,
                teacher_force=0.5):
        seq_len, batch_size = target.shape[0] - 1, target.shape[1]
        outputs = torch.zeros(seq_len, batch_size, self.vocab_size)
        decoder_hiddens = torch.zeros(seq_len, batch_size, self.hidden_size)
        source_mask = create_mask(source, self.input_vocab)

        word = target[0]
        h = init_hidden
        attention = torch.zeros(seq_len, batch_size, encoder_hiddens.shape[0])
        tf_coin = (random.random() < teacher_force)
        for i in range(seq_len):
            y, h, a = self.forward_step(word, h, encoder_hiddens, source_mask)
            attention[i] = a
            outputs[i] = y
            decoder_hiddens[i] = h
            if tf_coin:
                word = target[i + 1]
            else:
                word = y.argmax(dim=1)
        return outputs, decoder_hiddens, attention

    def forward_step(self, word, hidden, encoder_hiddens, source_mask):
        embedded_word = self.embedding(word)
        rnn_input = F.relu(embedded_word)
        if self.attention:
            a = self.attention(encoder_hiddens, hidden, source_mask)
            # a = [batch size, src len]
            a = a.unsqueeze(1)
            # a = [batch size, 1, src len]
            encoder_hiddens = encoder_hiddens.permute(1, 0, 2)
            # encoder_hiddens = [batch size, src len, enc hid dim]
            weighted_encoder_rep = torch.bmm(a, encoder_hiddens)
            # weighted_encoder_rep = [batch size, 1, enc hid dim]
            weighted_encoder_rep = weighted_encoder_rep.squeeze(1)
            # weighted_encoder_rep = [batch size, enc hid dim]
            rnn_input = torch.cat((rnn_input, weighted_encoder_rep), dim=1)
        # print(rnn_input.shape, hidden.shape)
        batch_size = rnn_input.shape[0]
        h = self.grucell(rnn_input, hidden.view(batch_size, -1))
        y = self.embedding_out(h)
        return y, h, a.squeeze(1)


class NullAttention(nn.Module):
    def __init__(self):
        super(NullAttention, self).__init__()

    def forward(self, encoder_hidden, decoder_hidden, source_mask):
        seq_len, batch_size = encoder_hidden.shape[0], encoder_hidden.shape[1]
        weights = torch.zeros(batch_size, seq_len)
        return weights


class PositionAttention(nn.Module):
    def __init__(self, decoder_size, max_seq_len=100):
        super(PositionAttention, self).__init__()
        self.decoder_size = decoder_size
        self.max_seq_len = max_seq_len
        self.attention_map = nn.Linear(self.decoder_size, self.max_seq_len)

    def forward(self, encoder_hidden, decoder_hidden, source_mask):
        seq_len, batch_size = encoder_hidden.shape[0], encoder_hidden.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                'input sequence too long to calculate positional attention')
        weights = self.attention_map(decoder_hidden).view(batch_size,
                                                          self.max_seq_len)
        weights = weights[:, :seq_len]
        weights[~source_mask] = -float("Inf")
        return F.softmax(weights, dim=1)


class MultiplicativeAttention(nn.Module):
    def __init__(self, decoder_size):
        super(MultiplicativeAttention, self).__init__()
        # We assume decoder and encoder hidden vectors are of the same size
        # (modulo bidirectionality)
        self.encoder_size = decoder_size
        self.decoder_size = decoder_size
        self.attention_map = None  # fill this in

    def forward(self, encoder_hidden, decoder_hidden, source_mask,
                sent_lengths=None):
        # encoder_hidden (as returned by pytorch's gru or lstm module) is of
        #  shape (seq_len, batch, num_directions, hidden_size)
        seq_len, batch_size = encoder_hidden.shape[0], encoder_hidden.shape[1]
        # first collapse across multiple directions (since our encoder is
        # unidirectional)
        encoder_hidden = encoder_hidden.view(seq_len, batch_size, -1)
        # next permute the encoder vectors so that batches is the 0th dimension
        encoder_hidden = encoder_hidden.permute(1, 0, 2)

        # Fill in your code here
        weights = None
        # weights should be of shape (batch, seq_len)
        return weights


class AdditiveAttention(nn.Module):
    def __init__(self, decoder_size, attention_size=None):
        super(AdditiveAttention, self).__init__()
        # Assuming decoder and encoder hidden vectors are of the same size (
        # modulo bidirectionality)
        self.decoder_size = decoder_size
        self.encoder_size = decoder_size
        # By default, the size of the attention vector is the same as the
        # decoder vector
        self.attention_size = attention_size if attention_size else \
            decoder_size
        self.encoder_map = None
        self.decoder_map = None
        self.v = None

    def forward(self, encoder_hidden, decoder_hidden, source_mask):
        # encoder_hidden (as returned by pytorch's gru or lstm module) is of
        #  shape (seq_len, batch, num_directions, hidden_size)
        seq_len, batch_size = encoder_hidden.shape[0], encoder_hidden.shape[1]
        # first collapse across multiple directions (since our encoder is
        # unidirectional)
        encoder_hidden = encoder_hidden.view(seq_len, batch_size, -1)
        # next permute the encoder vectors so that batches is the 0th dimension
        encoder_hidden = encoder_hidden.permute(1, 0, 2)

        # Fill in your code here

        weights = None
        # weights should be of shape (batch, seq_len)
        return weights


def create_mask(source, vocab):
    source = source.T
    max_len = source.shape[1]
    mask = torch.zeros_like(source, dtype=torch.uint8)
    eos_indices = (source == vocab.stoi['<eos>']).nonzero()
    lengths = eos_indices[:, 1] - 1  # -1 removes transformation as well as eos
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask[:, 0] = False
    return mask
