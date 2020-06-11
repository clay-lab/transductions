import torch.nn as nn


# TODO: DO WE USE THE TREE ENCODER/DECODER ANYMORE?
# ENCODERS FROM MODELS.PY, MODELSNEW.PY, BOB_MODELS.PY, MODELSNEWBOB.PY (MASTER), AND MODELSNEWBOB.PY (NEW-MODELS)
# Generic sequential encoder
class EncoderRNN(nn.Module): # FROM MODELS.PY --MASTER
    def __init__(self, vocab_size, hidden_size, recurrent_unit, n_layers=1, max_length=30, dropout_p=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout_p)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        else:
                print("Invalid recurrent unit type")


    # For succesively generating each new output and hidden layer
    def forward(self, batch):

        #outputs = Variable(torch.zeros(self.max_length, batch_size, self.hidden_size))
        #outputs = outputs.to(device=available_device) # to be used by attention in the decoder
        embedded_source = self.dropout(self.embedding(batch))
        outputs, final_hiddens = self.rnn(embedded_source)
        final_output = outputs[-1]
        #only return the last timestep's h vectors for the last encoder layer
        final_hiddens = final_hiddens[-1]

        return final_output, final_hiddens, outputs

# Generic sequential encoder
class EncoderRNN(nn.Module): # FROM MODELSNEW.PY -- MASTER
    def __init__(self, vocab_size, hidden_size, recurrent_unit, n_layers=1, max_length=30, dropout_p=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = n_layers
        self.hidden_size = hidden_size
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout_p)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = self.num_layers, dropout = dropout_p)
        else:
                print("Invalid recurrent unit type")


    # For succesively generating each new output and hidden layer
    def forward(self, batch):

        #outputs = Variable(torch.zeros(self.max_length, batch_size, self.hidden_size))
        #outputs = outputs.to(device=available_device) # to be used by attention in the decoder
        embedded_source = self.dropout(self.embedding(batch))
        outputs, final_hiddens = self.rnn(embedded_source)
        final_output = outputs[-1]
        #only return the last timestep's h vectors for the last encoder layer
        final_hiddens = final_hiddens[-1]

        return final_output, final_hiddens, outputs

class Encoder(nn.Module): # FROM BOB_MODELS.PY -- MASTER
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

# Generic sequential encoder
class EncoderRNN(nn.Module): # FROM MODELSNEWBOB.PY -- MASTER
    def __init__(self, vocab_size, hidden_size, recurrent_unit, num_layers=1, max_length=30, dropout=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        else:
                print("Invalid recurrent unit type")

    def forward(self, batch):

        embedded_source = self.dropout(self.embedding(batch))
        outputs, final_hiddens = self.rnn(embedded_source)
        if self.rnn_type == 'LSTM':
            final_hiddens = final_hiddens[0] # ignore cell state
        final_output = outputs[-1]
        #only return the last timestep's h vectors for the last encoder layer 
        #Note that doing this should make the values of final_output and final_hiddens the same!
        final_hiddens = final_hiddens[-1]

        return final_output, final_hiddens, outputs

# Generic sequential encoder
class EncoderRNN(nn.Module): # FROM MODELSNEWBOB.PY -- NEW-MODELS BRANCH
    def __init__(self, vocab_size, hidden_size, recurrent_unit, num_layers=1, max_length=30, dropout=0):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_type = recurrent_unit
        self.max_length = max_length
        self.dropout = nn.Dropout(p=dropout)

        self.embedding = nn.Embedding(vocab_size, hidden_size)


        if num_layers == 1: dropout = 0 
        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers = num_layers, dropout = dropout)
        else:
                print("Invalid recurrent unit type")

    def forward(self, batch):

        embedded_source = self.dropout(self.embedding(batch))
        outputs, state = self.rnn(embedded_source)
        #final_output = outputs[-1]
        #only return the h (and c) vectors for the last encoder layer 
        #if self.rnn_type == 'LSTM':
        #    final_hiddens, final_cell = state 
        #    state = (final_hiddens[-1], final_cell[-1]) #take the last layer of hidden and cell
        #else:
        #    state = state[-1] #take the last layer of hidden (for GRU and SRN)
        return state, outputs



