from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

import torchtext.data as tt
import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from functools import reduce

from model import Seq2Seq
from data_prep import prepare_csv, load_and_prepare_dataset
from early_stopping import EarlyStopping

def train_epoch(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          short_train: bool = False,
          teacher_force: float = 0.5):
    model.train()
    epoch_loss = 0.
    n_batches = len(iterator)
    for n, batch in enumerate(iterator):
        if short_train and n % 20 != 0:
            continue
        src, trg = batch.src, batch.trg
        batch_loss = torch.tensor(0., requires_grad=True)
        optimizer.zero_grad()
        output = model(src, trg, teacher_force)
        for i in range(trg.shape[0]-1):
            batch_loss = batch_loss + criterion(output[i],trg[i+1])
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += batch_loss.item()
        #if (n % 200) == 0:
        #   print("batch", n, "/", n_batches, "batch loss/word", (batch_loss/trg.shape[0]).item())
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module,
             eval_words=None):
    model.eval()
    epoch_loss = 0.
    total_preds = 0.
    correct_preds = 0.
    pad_index = model.output_vocab.stoi['<pad>']
    if eval_words == None:
        eval_words = list(range(len(model.output_vocab.itos)))
        eval_words.remove(pad_index)
    else: 
        eval_words = [model.output_vocab.stoi[x] for x in eval_words]
    with torch.no_grad():
        for batch in iterator:
            src, trg = batch.src, batch.trg
            output = model(src, trg, 0) #turn off teacher forcing
            for i in range(trg.shape[0]-1):
                epoch_loss = epoch_loss + criterion(output[i],trg[i+1]).item()
                eval_locations = reduce(lambda a, b: a | b, [(trg[i+1] == x) for x in eval_words]) 
                total_preds += sum(eval_locations)
                correct_preds += sum(eval_locations & (output[i].argmax(dim=1) == trg[i+1])) 
    return (epoch_loss / len(iterator)), (correct_preds / total_preds) * 100.


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, train_iterator, valid_iterator, test_iterator, optimizer, criterion, clip=1, short_train=True, n_epochs=10, teacher_force=0.5, eval_words=None, patience=3):
    early_stopping = EarlyStopping(patience=patience, verbose=False, filename='cache/checkpoint.pt')
    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train_epoch(model, train_iterator, optimizer, criterion, clip, short_train, teacher_force=teacher_force)
        valid_loss, valid_accuracy = evaluate(model, valid_iterator, criterion, eval_words = eval_words)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3E}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3E}')
        print(f'\t Val. Accuracy: {valid_accuracy:.3f}')
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping, reloading checkpoint model")
            model.load_state_dict(torch.load('cache/checkpoint.pt'))
            break

    test_loss, test_accuracy = evaluate(model, test_iterator, criterion, eval_words=eval_words)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3E} |')
    print(f'| Test Accuracy: {test_accuracy:.3f}')

def save_model(model):
    torch.save(model.state_dict(), 
    model.attention_string+"E"+str(model.encoder.embedding_size)+"H"+str(model.encoder.hidden_size)+".pt")

def load_model(PATH:str, src_text, trg_text):
    attention = PATH[:PATH.index('E')]
    embedding_size = int(PATH[PATH.index('E')+1:PATH.index('H')])
    hidden_size = int(PATH[PATH.index('H')+1:PATH.index('.')])
    model = Seq2Seq(src_text, embedding_size, hidden_size, trg_text, attention)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model

if __name__ == "__main__":
    prepare_csv('negation')    
    train_iterator, valid_iterator, test_iterator, src_text, trg_text = load_and_prepare_dataset('negation', 100)
    # ,trans_text
    PAD_IDX = trg_text.vocab.stoi['<pad>']

    EMBEDDING_SIZE = 32
    HIDDEN_SIZE = 64
    
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    model = Seq2Seq(src_text, EMBEDDING_SIZE, HIDDEN_SIZE, trg_text, attention='Position')
    optimizer = optim.Adam(model.parameters())

    N_EPOCHS = 10
    train(model, train_iterator, val_iterator, val_iterator, optimizer, criterion, short_train=False, n_epochs=N_EPOCHS)
    #save_model(model)
        