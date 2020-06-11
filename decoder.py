# DECODERS FROM MODELS.PY, MODELSNEW.PY, BOB_MODELS.PY, MODELSNEWBOB.PY (MASTER), AND MODELSNEWBOB.PY (NEW-MODELS)
# Generic sequential decoder
class DecoderRNN(nn.Module): # FROM MODELS.PY -- MASTER
    def __init__(self, hidden_size, output_size, recurrent_unit, attn=False, n_layers=1, dropout_p=0.1, max_length=30):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attn

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "SquashedLSTM":
                self.rnn = SquashedLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "UnsquashedGRU":
                self.rnn = UnsquashedGRU(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.recurrent_unit = recurrent_unit

        # location-based attention
        if attn == "location":
                # Attention vector
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

                # Context vector made by combining the attentions
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # content-based attention
        if attn == "content": 
                self.v = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
                nn.init.uniform(self.v, -1, 1) # maybe need cuda
                self.attn_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    # Perform one step of the forward pass
    def forward_step(self, input, hidden, encoder_outputs, input_variable):
        output = self.embedding(input).unsqueeze(0)
        output = self.dropout(output)

        attn_weights = None

        batch_size = input_variable.size()[1]

        # Determine attention weights using location-based attention
        if self.attention == "location":
                if self.recurrent_unit == "LSTM" or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)))
                else:
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)))

                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        # Determine attention weights using content-based attention
        if self.attention == "content": 
                input_length = input_variable.size()[0] 
                u_i = Variable(torch.zeros(len(encoder_outputs), batch_size))

                u_i = u_i.to(device=available_device)


                for i in range(input_length):
                        if self.recurrent_unit == "LSTM"  or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0][0].unsqueeze(0), output), 2)))
                        else:
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0].unsqueeze(0), output), 2)))
                        u_i_j = torch.bmm(attn_hidden, self.v.unsqueeze(1).unsqueeze(0))
                        u_i[i] = u_i_j[0].view(-1)


                a_i = F.softmax(u_i.transpose(0,1)) 
                attn_applied = torch.bmm(a_i.unsqueeze(1), encoder_outputs.transpose(0,1))

                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    # Perform the full forward pass
    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5, evaluation=False):
        input_variable = training_set[0]
        target_variable = training_set[1]

        batch_size = training_set[0].size()[1]

        decoder_input = Variable(torch.LongTensor([0] * batch_size))
        decoder_input = decoder_input.to(device=available_device)

        decoder_hidden = hidden
        
        decoder_outputs = []

        use_tf = True if random.random() < tf_ratio else False

        if use_tf: # Using teacher forcing
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                                decoder_input, decoder_hidden, encoder_outputs, input_variable)
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output)

        else: # Not using teacher forcing
            if evaluation:
                end_num = 100
            else:
                end_num = target_variable.size()[0]

            for di in range(end_num): 
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                            decoder_input, decoder_hidden, encoder_outputs, input_variable) 

                topv, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi.view(-1))
                decoder_input = decoder_input.to(device=available_device)

                decoder_outputs.append(decoder_output)

                if 1 in topi[0] or 2 in topi[0]:
                    break

        return decoder_outputs 

# Generic sequential decoder
class DecoderRNN(nn.Module): # FROM MODELSNEW.PY
    def __init__(self, hidden_size, output_size, recurrent_unit, attn=False, n_layers=1, dropout_p=0.1, max_length=30):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.attention = attn

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)

        if recurrent_unit == "SRN":
                self.rnn = nn.RNN(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRU(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "SquashedLSTM":
                self.rnn = SquashedLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "ONLSTM":
                self.rnn = ONLSTM(self.hidden_size, self.hidden_size)
        elif recurrent_unit == "UnsquashedGRU":
                self.rnn = UnsquashedGRU(hidden_size, hidden_size)
        else:
                print("Invalid recurrent unit type")

        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.recurrent_unit = recurrent_unit

        # location-based attention
        if attn == "location":
                # Attention vector
                self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

                # Context vector made by combining the attentions
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # content-based attention
        if attn == "content": 
                self.v = nn.Parameter(torch.FloatTensor(hidden_size), requires_grad=True)
                nn.init.uniform(self.v, -1, 1) # maybe need cuda
                self.attn_layer = nn.Linear(self.hidden_size * 3, self.hidden_size)
                self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

    # Perform one step of the forward pass
    def forward_step(self, input, hidden, encoder_outputs, input_variable):
        output = self.embedding(input).unsqueeze(0)
        output = self.dropout(output)

        attn_weights = None

        batch_size = input_variable.size()[1]

        # Determine attention weights using location-based attention
        if self.attention == "location":
                if self.recurrent_unit == "LSTM" or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0][0]), 1)))
                else:
                    attn_weights = F.softmax(self.attn(torch.cat((output[0], hidden[0]), 1)))

                attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0,1))
                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        # Determine attention weights using content-based attention
        if self.attention == "content": 
                input_length = input_variable.size()[0] 
                u_i = Variable(torch.zeros(len(encoder_outputs), batch_size))

                u_i = u_i.to(device=available_device)


                for i in range(input_length):
                        if self.recurrent_unit == "LSTM"  or self.recurrent_unit == "ONLSTM" or self.recurrent_unit == "SquashedLSTM":
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0][0].unsqueeze(0), output), 2)))
                        else:
                                attn_hidden = F.tanh(self.attn_layer(torch.cat((encoder_outputs[i].unsqueeze(0), hidden[0].unsqueeze(0), output), 2)))
                        u_i_j = torch.bmm(attn_hidden, self.v.unsqueeze(1).unsqueeze(0))
                        u_i[i] = u_i_j[0].view(-1)


                a_i = F.softmax(u_i.transpose(0,1)) 
                attn_applied = torch.bmm(a_i.unsqueeze(1), encoder_outputs.transpose(0,1))

                attn_applied = attn_applied.transpose(0,1)

                output = torch.cat((output[0], attn_applied[0]), 1)
                output = self.attn_combine(output).unsqueeze(0)

        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    # Perform the full forward pass
    def forward(self, hidden, encoder_outputs, training_set, tf_ratio=0.5, evaluation=False):
        input_variable = training_set[0]
        target_variable = training_set[1]

        batch_size = training_set[0].size()[1]

        decoder_input = Variable(torch.LongTensor([0] * batch_size))
        decoder_input = decoder_input.to(device=available_device)

        decoder_hidden = hidden
        
        decoder_outputs = []

        use_tf = True if random.random() < tf_ratio else False

        if use_tf: # Using teacher forcing
            for di in range(target_variable.size()[0]):
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                                decoder_input, decoder_hidden, encoder_outputs, input_variable)
                decoder_input = target_variable[di]
                decoder_outputs.append(decoder_output)

        else: # Not using teacher forcing
            if evaluation:
                end_num = 100
            else:
                end_num = target_variable.size()[0]

            for di in range(end_num): 
                decoder_output, decoder_hidden, decoder_attention = self.forward_step(
                            decoder_input, decoder_hidden, encoder_outputs, input_variable) 

                topv, topi = decoder_output.data.topk(1)
                decoder_input = Variable(topi.view(-1))
                decoder_input = decoder_input.to(device=available_device)

                decoder_outputs.append(decoder_output)

                if 1 in topi[0] or 2 in topi[0]:
                    break

        return decoder_outputs 


class Decoder(nn.Module): # FROM BOB_MODELS.PY
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


# Generic sequential decoder
class DecoderRNN(nn.Module): # MODELSNEWBOB.PY -- MASTER
    def __init__(self, hidden_size, vocab, encoder_vocab, recurrent_unit, embedding_size=None, attention_type=None, num_layers=1, dropout=0, max_length=30):
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

        if recurrent_unit == "SRN":
                self.rnn = nn.RNNCell(self.embedding_size + (hidden_size if attention_type else 0), 
                                      hidden_size)
        elif recurrent_unit == "GRU":
                self.rnn = nn.GRUCell(self.embedding_size + (hidden_size if attention_type else 0),
                                      hidden_size)
        elif recurrent_unit == "LSTM":
                self.rnn = nn.LSTMCell(self.embedding_size + (hidden_size if attention_type else 0),
                                      hidden_size)
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
            a = self.attn(encoder_outputs, hidden, source_mask) 
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
        state = self.rnn(rnn_input, h)
        h = state[0] if isinstance(state,tuple) else state #For LSTM pass only h to output
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
        else: 
            #if we're doing teacher forcing, don't generate past the length of the target
            gen_length = target.shape[0]
        
        source_mask = create_mask(source, self.encoder_vocab)
        
        #initialize x and h to given initial values. 
        # Assumes that the first position in target is <bos> and can be ignored.
        x, h = torch.tensor([self.vocab.stoi[x] for x in x0]), h0
        output_complete_flag = torch.zeros(batch_size, dtype=torch.bool)
        for i in range(gen_length): 
            if self.recurrent_unit_type == "LSTM":
                if not(isinstance(h0,tuple)):
                    c0 = torch.zeros(batch_size, self.hidden_size)
                    h = (h0, c0)
            y, h, a = self.forwardStep(x, h, encoder_outputs, source_mask)
            outputs[i] = y
            attention[i] = a
            decoder_hiddens[i] = h if self.recurrent_unit_type is not "LSTM" else h[0]
            if (evaluation | (random.random() > tf_ratio)):
                x = y.argmax(dim=1)
            else:
                x = target[i]  
            #stop if all of the examples in the batch include eos or pad
            output_complete_flag += ((x == self.eos_index) | (x == self.pad_index))
            if all(output_complete_flag):
                break
        return outputs[:gen_length]#, decoder_hiddens[:i+1], attention[:i+1]


# Generic sequential decoder
class DecoderRNN(nn.Module): # FROM MODELSNEWBOB.PY IN MODELS-NEW BRANCH
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



