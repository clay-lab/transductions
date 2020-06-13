import numpy as np
import torch

"""
assume encoder and decoder outputs do NOT have softmax yet
"""
"""
output from decoder should always be of shape: max sequence length x batch size x size of vocab
target tensor should be of shape: max sequence length x batch size
"""
class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, encoder_field_names, decoder_field_names, encoder_train_field_names=None, decoder_train_field_names=None, middle_field_name="middle"):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.encoder_field_names = encoder_field_names
        self.decoder_field_names = decoder_field_names
        
        self.encoder_train_field_names = self.encoder_field_names if (encoder_train_field_names is None) else encoder_train_field_names
        self.decoder_train_field_names = self.decoder_field_names if (decoder_train_field_names is None) else decoder_train_field_names
        
        self.middle_field_name = middle_field_name

    @classmethod
    def from_cmd_line_args(cls, encoder_name, decoder_name):
        assert False

    def forward(self, batch):
        enc_fields = self.encoder_train_field_names if self.training else self.encoder_field_names
        dec_fields = self.decoder_train_field_names if self.training else self.decoder_field_names

        middle = self.encoder(*(getattr(batch, fieldname) for fieldname in enc_fields))

        # TODO: this is a dirty dirty hack and ought to be replaced
        if isinstance(middle, tuple):
            # encoder has multiple outputs, which each need a name
            for ix, entry in enumerate(middle):
                setattr(batch, self.middle_field_name+str(ix), entry)
        else:
            # enocder has one output
            setattr(batch, self.middle_field_name, middle)

        return self.decoder(*(getattr(batch, fieldname) for fieldname in dec_fields))

        # delattr(batch, self.middle_field_names)

    def scores2sentence(self, scores, vocab):
        ids = scores.transpose(0, 1).contiguous().view(-1)
        tests = np.reshape([vocab.itos[i] for i in ids], tuple(scores.size())[::-1])
        tests = [' '.join(r) for r in tests]
        return tests
        # return scores.apply_(f)
        # ix2word = np.array(vocab.itos)
        # word_ixs = scores.argmax(dim=2)
        # word_ixs = word_ixs.detach().numpy()
        # return ix2word[word_ixs]

    def to_sentence(self, batch):
        assert len(batch.target_fields) == 1
        target_vocab = batch.dataset.fields[batch.target_fields[0]].vocab

        output = self(batch)    # shape: sequence length x batch size x vocab size
        pred_words = output.argmax(2)

        return reverse_tokenization(pred_words, target_vocab)


def reverse_tokenization(batch_ixs, vocab):
    ix2word = np.array(vocab.itos)
    
    if isinstance(batch_ixs, torch.Tensor):
        batch_ixs = batch_ixs.detach().numpy()
        
    words = ix2word[batch_ixs]
    
    sentences = [" ".join(words[:, eg]) for eg in range(words.shape[1])]
    
    return sentences