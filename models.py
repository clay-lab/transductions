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
        
# FROM MODELSNEWBOB.PY IN MODELS-NEW BRANCH
# Generic sequential decoder
class DecoderRNN(nn.Module): 
    def __init__(self, hidden_size, vocab, encoder_vocab, recurrent_unit, embedding_size=None, attention_type=None, num_layers=1, dropout=0.1, max_length=30):
        super(DecoderRNN, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.eos_index = self.vocab.stoi['<eos>']
        self.pad_index = self.vocab.stoi['<pad>']
        self.encoder_vocab = encoder_vocab
        self.hidden_size = hidden_size
        self.embedding_size = hidden_size if embedding_size == None else embedding_size
        self.embedding = nn.Embedding(self.vocab_size, hidden_size)
        self.num_layers = num_layers
        self.max_length = max_length
        self.attention_type = attention_type
        self.recurrent_unit_type = recurrent_unit

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.hidden_size, self.vocab_size)

        if num_layers == 1: dropout = 0 
        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.embedding_size + (hidden_size if attention_type else 0), 
                                      hidden_size, num_layers=num_layers, dropout=dropout)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.embedding_size + (hidden_size if attention_type else 0),
                                      hidden_size, num_layers=num_layers, dropout=dropout)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.embedding_size + (hidden_size if attention_type else 0),
                                      hidden_size, num_layers=num_layers, dropout=dropout)
        else:
                print("Invalid recurrent unit type")

        # location-based attention
        if attention_type == "location":
                self.attn = PositionAttention(hidden_size, max_length = self.max_length)
        # additive/content-based (Bahdanau) attention
        if attention_type == "additive": 
                self.attn = AdditiveAttention(hidden_size)
        #multiplicative (key-value) attention
        if attention_type == "multiplicative":
                self.attn = MultiplicativeAttention(hidden_size)
        #dot product attention
        if attention_type == "dotproduct":
                self.attn = DotproductAttention()

    def forwardStep(self, x, h, encoder_outputs, source_mask):
        x = self.embedding(x)
        #Apply ReLU to embedded input?
        rnn_input = F.relu(x)
        
        if self.attention_type:
            #use h alone for attention key in case we're dealing with LSTM
            if isinstance(h,tuple):
                hidden = h[0]
            else:
                hidden = h
            #only give last layer's hidden state to attention
            a = self.attn(encoder_outputs, hidden[-1], source_mask) 
            #attn_weights = [batch size, src len]
            a = a.unsqueeze(1)
            #a = [batch size, 1, src len]
            encoder_outputs = encoder_outputs.permute(1,0,2)
            #encoder_hiddens = [batch size, src len, enc hid dim]
            weighted_encoder_outputs = torch.bmm(a, encoder_outputs)
            #weighted_encoder_outputs = [batch size, 1, enc hid dim]
            weighted_encoder_outputs = weighted_encoder_outputs.squeeze(1)
            #weighted_encoder_rep = [batch size, enc hid dim]
            rnn_input = torch.cat((rnn_input, weighted_encoder_outputs), dim=1)
        else:
             a = torch.zeros(encoder_outputs.shape[1], 1, encoder_outputs.shape[0])
        
        batch_size = rnn_input.shape[0] 
        _, state = self.rnn(rnn_input.unsqueeze(0), h)
        #Only include last h in output computation. for LSTM pass only h (not c)
        h = state[0][-1] if isinstance(state,tuple) else state[-1] 
        y = self.out(h)
        return y, state, a.squeeze(1)

    # Perform the full forward pass
    def forward(self, h0, x0, encoder_outputs, source, target=None, tf_ratio=0.5, evaluation=False):
        batch_size = encoder_outputs.shape[1]
        outputs = torch.zeros(self.max_length, batch_size, self.vocab_size)
        decoder_hiddens = torch.zeros(self.max_length, batch_size, self.hidden_size)
        attention = torch.zeros(self.max_length, batch_size, encoder_outputs.shape[0])

        #if we're evaluating, never use teacher forcing
        if (evaluation or not(torch.is_tensor(target))):
            tf_ratio=0.0
            gen_length = self.max_length
        #if we're doing teacher forcing, don't generate past the length of the target
        else: 
            gen_length = target.shape[0]
        
        source_mask = create_mask(source, self.encoder_vocab)
        #initialize x and h to given initial values. 
        x, h = torch.tensor([self.vocab.stoi[x] for x in x0]), h0
        #print('x before', x, x0, self.vocab.stoi)
        output_complete_flag = torch.zeros(batch_size, dtype=torch.bool)
        if self.recurrent_unit_type == "LSTM": #Non-LSTM encoder, LSTM decoder: create c
                if not(isinstance(h0,tuple)):
                    c0 = torch.zeros(self.num_layers,batch_size, self.hidden_size)
                    h = (h0, c0)
        elif isinstance(h0,tuple): #LSTM encoder, but not LSTM decoder: ignore c
            h = h[0]
        for i in range(gen_length): 
            y, h, a = self.forwardStep(x, h, encoder_outputs, source_mask)
            outputs[i] = y
            attention[i] = a
            decoder_hiddens[i] = h[-1] if self.recurrent_unit_type is not "LSTM" else h[0][-1]
            #print('y shape', y.shape, 'target shape', target.shape)
            if (evaluation | (random.random() > tf_ratio)):
                x = y.argmax(dim=1)
            else:
                x = target[i]  
            #stop if all of the examples in the batch include eos or pad
            #print('x after', x, output_complete_flag)
            output_complete_flag += ((x == self.eos_index) | (x == self.pad_index))
            if all(output_complete_flag):
                break
        return outputs[:gen_length]#, decoder_hiddens[:i+1], attention[:i+1]