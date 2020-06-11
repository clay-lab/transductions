
import os.path

import torch
from torch import optim
from tqdm import tqdm

from nltk.parse import ViterbiParser
import nltk.grammar

import RPNTask

import models




GRAMMAR = RPNTask.pre_arithmetic_grammar
OP_TAGS = ["EXPR", "START"]
vocab = RPNTask.get_terminals(GRAMMAR)
vocab.insert(0, '_')
index2word = vocab
word2index = {word: ix for ix, word in enumerate(vocab)}


def generate_training_pair(min_length, max_length, batching=True):
    parser = ViterbiParser(GRAMMAR)
    toks = RPNTask.draw_from_PCFG(GRAMMAR, min_length=min_length, max_length=max_length)
    pre_tree = parser.parse_one(toks)
    post_tree = RPNTask.polish2reversePolish(pre_tree, OP_TAGS)
    pre_tree.chomsky_normal_form()
    pre_tree.collapse_unary(True, True)
    post_tree.collapse_unary(True, True)
    pre_tree.__prob = -1 # for printing to work
    post_tree.__prob = -1 #for printint to work
    
    inseq = RPNTask.tree2seq(pre_tree, "none")
    inseq = torch.LongTensor([word2index[w] for w in inseq]) # no batching
    outseq = RPNTask.tree2seq(post_tree)
    outseq = torch.LongTensor([word2index[w] for w in outseq]) # no batching
    if batching:
        inseq = inseq.unsqueeze(1)
        # outseq = outseq.unsqueeze(0)    
    
    #return inseq, outseq, pre_tree, post_tree
    return inseq, outseq, RPNTask.tree2index_form(pre_tree), post_tree

################
# TRAINING FUNS
################

def train(training_pair, seq2seq, opter):
    seq2seq.train()
    opter.zero_grad()

    decoder_outputs = seq2seq(training_pair)
    
    weight = torch.ones(len(word2index))
    weight[word2index['_']] = topology_weight
    Loss = torch.nn.NLLLoss(weight)
    loss = Loss(decoder_outputs, training_pair[1])
    loss.backward()

    opter.step()
    
    return loss

def evaluate(seq2seq, total_sents, max_length, verbose=False, eval_mode=True, min_length=5):
    if eval_mode:
        seq2seq.eval()
    else:
        seq2seq.train()
    
    correct_tokens = 0
    total_tokens = 0
    correct_sents = 0
    correct_length = 0
    # for _ in tqdm(range(total_sents)):
    for _ in range(total_sents):
        training_minibatch = generate_training_pair(min_length, max_length)
    
        decoder_outputs = seq2seq(training_minibatch)
    
        pred = decoder_outputs.argmax(1)
        target = training_minibatch[1]
        if not seq2seq.training:
            target = target[target != seq2seq.decoder.null_ix] # in eval mode, we want to get rid of the nulls
        pred_target = torch.nn.utils.rnn.pad_sequence([pred, target], padding_value=-1)
        comp = (pred_target[:,0] == pred_target[:,1])
        total_tokens += len(comp)
        correct_tokens += torch.sum(comp).item()
        correct_sents += all(comp)
        correct_length += (len(pred) == len(target))
        
        if verbose:
            print("input:\t", " ".join(index2word[ix] for ix in training_minibatch[0]))
            print("target:\t", " ".join(index2word[ix] for ix in target))
            print("pred:\t", " ".join(index2word[ix] for ix in pred))
            print("\n\n")
        
    return {
        'total_sentences': total_sents,
        'total_tokens': total_tokens,
        'correct_sents': correct_sents,
        'correct_tokens': correct_tokens,
        'correct_length': correct_length,
        'sentence_acc': correct_sents / total_sents,
        'token_acc': correct_tokens / total_tokens,
        'length_acc': correct_length / total_sents
    }



# TRAINING TIME
if __name__ == "__main__":

    encoder_type = "GRU"
    encoder_layers = 1
    learning_rate = 0.05
    n_iters = 200_000
    eval_every = 10_000
    n_eval = 30
    topology_weight = 1
    hidden_size = 60 # guess
    MAX_LENGTH = 15
    MIN_LENGTH = 5
    savefile = "seq2seq_5_22.pt"

    encoder = models.EncoderRNN(len(word2index), int(hidden_size), encoder_type, max_length=MAX_LENGTH, n_layers=encoder_layers)
    #encoder = models.TreeEncoderRNNNew(len(word2index), int(hidden_size)) 
    decoder = models.GRUTridentDecoder(3, len(word2index.keys()) - 1, int(hidden_size), 5)
    #decoder = models.TridentDecoder(3, len(word2index) - 1, int(hidden_size), 5) # -1 b/c word2index already includes null
    seq2seq = models.Seq2Seq(encoder, decoder)

    opter = optim.SGD(seq2seq.parameters(), lr=learning_rate)

    if os.path.exists(savefile):
        seq2seq.load_state_dict(torch.load(savefile))
        print("Load success!")

    total_train_loss = torch.tensor(0.)
    subiter = 0
    with tqdm(range(n_iters)) as t:
        for it in t:
                
            t.set_postfix(avg_train_loss=total_train_loss.item()/(subiter + 1))

            training_minibatch = generate_training_pair(MIN_LENGTH, MAX_LENGTH)

            total_train_loss += train(
                training_minibatch, seq2seq, opter)

            subiter += 1
            
            if it % eval_every == 0 and it > 0:

                stats = evaluate(seq2seq, n_eval, MAX_LENGTH)
                evaluation_msg = f"Dev performance after {it} minibatches\n\tSentence accuracy: {100*stats['sentence_acc']:.2f}%\n\tToken accuracy {100*stats['token_acc']:.2f}%\n\tLength accuracy {100*stats['length_acc']:.2f}%"
                print(evaluation_msg)

                torch.save(seq2seq.state_dict(), savefile)
                
                total_train_loss = torch.tensor(0.)
                subiter = 0