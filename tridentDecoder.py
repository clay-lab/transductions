# TODO: DO WE USE THE OTHER TRIDENT DECODERS?
class TridentDecoder(nn.Module): # FROM MODELS.PY
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(TridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        self.to_children = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.arity * self.hidden_size))

    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, tree=None):
        batch_size = root_hidden.shape[0]
        batch_outs = []
        for eg_ix in range(batch_size):
            batch_outs.append(self.forward_nobatch(root_hidden[eg_ix, :], tree=(None if tree is None else tree[eg_ix])))

        return nn.utils.rnn.pad_sequence(batch_outs)

    def forward_nobatch(self, root_hidden, tree=None):
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            assert tree is not None
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            #return F.log_softmax(raw_scores, dim=1)
            return raw_scores

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_train_helper(children_hidden[child_ix], root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)

        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_eval_helper(children_hidden[child_ix], depth+1)
        else:
            yield production

    def _hidden2children(self, hidden):
        return torch.split(self.to_children(hidden), self.hidden_size, dim=-1)

class TridentDecoder(nn.Module): # FROM MODELSNEW.PY
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(TridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        self.to_children = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.arity * self.hidden_size))

    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, tree=None):
        batch_size = root_hidden.shape[0]
        batch_outs = []
        for eg_ix in range(batch_size):
            batch_outs.append(self.forward_nobatch(root_hidden[eg_ix, :], tree=(None if tree is None else tree[eg_ix])))

        return nn.utils.rnn.pad_sequence(batch_outs)

    def forward_nobatch(self, root_hidden, tree=None):
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            assert tree is not None
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            #return F.log_softmax(raw_scores, dim=1)
            return raw_scores

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_train_helper(children_hidden[child_ix], root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)

        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_eval_helper(children_hidden[child_ix], depth+1)
        else:
            yield production

    def _hidden2children(self, hidden):
        return torch.split(self.to_children(hidden), self.hidden_size, dim=-1)

class TridentDecoder(nn.Module): # MODELSNEWBOB.PY -- MASTER
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(TridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        self.to_children = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.arity * self.hidden_size))

    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, training_set):
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            tree = training_set[3]
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            return F.log_softmax(raw_scores, dim=1)

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_train_helper(children_hidden[child_ix], root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)

        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_eval_helper(children_hidden[child_ix], depth+1)
        else:
            yield production

    def _hidden2children(self, hidden):
        return torch.split(self.to_children(hidden), self.hidden_size, dim=-1)


class TridentDecoder(nn.Module): # FROM MODELSNEWBOB.PY NEW-MODELS BRANCH
    def __init__(self, arity, vocab_size, hidden_size, max_depth, null_placement="pre"):
        super(TridentDecoder, self).__init__()

        self.arity = arity
        self.null_placement = null_placement

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size + 1 # first one is for null
        self.max_depth = max_depth

        self.hidden2vocab = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.vocab_size))
        self.to_children = nn.Sequential(nn.Linear(self.hidden_size, 2*self.hidden_size), nn.Sigmoid(), nn.Linear(2*self.hidden_size, self.arity * self.hidden_size))

    @property
    def null_ix(self):
        return 0

    def forward(self, root_hidden, training_set):
        #If the root_hidden input is from an LSTM, take h
        if isinstance(root_hidden,tuple):
            root_hidden = root_hidden[0]
        #only take the last hidden layer from the encoder
        root_hidden = root_hidden[-1, :, :].squeeze(1)
        print('root_hidden', root_hidden.shape)
        
        if self.training:
            # assumption: every non-terminal node either has 3 children which are all non-terminals, or is the parent of one terminal node
            tree = training_set[3]
            raw_scores = torch.stack(list(self.forward_train_helper(root_hidden, tree)))
            return F.log_softmax(raw_scores, dim=1)
        else:
            raw_scores = torch.stack(list(self.forward_eval_helper(root_hidden)))
            return F.log_softmax(raw_scores, dim=1)

    def forward_train_helper(self, root_hidden, root):
        production = self.hidden2vocab(root_hidden)
        if len(root) > 1:
            assert len(root) == self.arity

            assert self.null_placement == "pre"
            yield production
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_train_helper(children_hidden[child_ix], root[child_ix])
        else:
            yield production

    def forward_eval_helper(self, root_hidden, depth=0):
        production = self.hidden2vocab(root_hidden)

        if (torch.argmax(production) == self.null_ix) and (depth <= self.max_depth):
            # we chose NOT to output a word here... recurse more
            children_hidden = self._hidden2children(root_hidden)
            for child_ix in range(self.arity):
                yield from self.forward_eval_helper(children_hidden[child_ix], depth+1)
        else:
            yield production

    def _hidden2children(self, hidden):
        return torch.split(self.to_children(hidden), self.hidden_size, dim=-1)
