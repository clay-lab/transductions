import torch.nn as nn

"""
assume encoder and decoder outputs do NOT have softmax yet
"""
class Seq2Seq(nn.Module):
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