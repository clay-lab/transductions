"""
assume encoder and decoder outputs do NOT have softmax yet
"""
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, encoder_field_names, decoder_field_names, encoder_train_field_names=None, decoder_train_field_names=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_field_names = encoder_field_names
        self.decoder_field_names = decoder_field_names
        
        self.encoder_train_field_names = self.encoder_field_names if (encoder_train_field_names is None) else encoder_train_field_names
        self.decoder_train_field_names = self.decoder_field_names if (decoder_train_field_names is None) else decoder_train_field_names
        
    @classmethod
    def from_cmd_line_args(cls, encoder_name, decoder_name):
        assert False

    def forward(self, example):
        enc_fields = self.encoder_train_field_names if self.training else self.encoder_field_names
        dec_fields = self.decoder_train_field_names if self.training else self.decoder_field_names

        encoding = self.encoder(*(getattr(example, fieldname) for fieldname in enc_fields))

        return self.decoder(encoding, *(getattr(example, fieldname) for fieldname in dec_fields))