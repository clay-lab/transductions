import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from early_stopping import EarlyStopping
from abc import abstractmethod
 
CKPT_NAME_LATEST = "latest_ckpt.pt"

class AverageMetric:

    def __init__(self):
        self.correct = 0
        self.total = 0

    @abstractmethod
    def process_batch(self, prediction, target):
        pass

    def update(self, correct, total=1):
        self.correct += correct
        self.total += total

    def result(self):
        return 1.0 * self.correct / self.total if self.total > 0 else np.nan

class SentenceLevelAccuracy(AverageMetric):

    def process_batch(self, prediction, target, model):  
        correct = (prediction == target).prod(axis=0)
        total = correct.size()[0]

        self.update(correct.sum(), total)

class TokenLevelAccuracy(AverageMetric):

    def process_batch(self, prediction, target, model): 
        correct = (prediction == target).sum()
        total = target.size()[0] * target.size()[1]

        self.update(correct, total)

class LengthLevelAccuracy(AverageMetric):

    def __init__(self):
        AverageMetric.__init__(self)
        self.total = 1

    def process_batch(self, prediction, target, model): 
        pass

def predict(model, source):

    # build batch from tensor

    model.eval()
    with torch.no_grad():

        logits = model(batch)
        predictions = logits[:source.size()[0], :].argmax(2)
        sentences = model.scores2sentence(predictions, model.decoder.vocab)

        return sentences

def test(model, test_iter, task, filename):

    model.eval()

    with open(filename, 'w') as f:
        f.write('{0}\t{1}\t{2}\n'.format('source', 'target', 'prediction'))
    with torch.no_grad():
        print("Testing on test data")
        with tqdm(test_iter) as t:
            for batch in t:

                logits = model(batch)
                target = batch.target 
                predictions = logits[:target.size()[0], :].argmax(2)

                sentences = model.scores2sentence(batch.source, model.encoder.vocab)
                predictions = model.scores2sentence(predictions, model.decoder.vocab)
                target = model.scores2sentence(target, model.decoder.vocab)

                with open(filename, 'a') as f:
                    for i, _ in enumerate(sentences):
                        f.write('{0}\t{1}\t{2}\n'.format(
                            sentences[i], target[i], predictions[i])
                        )

def evaluate(model, val_iter, epoch, args, criterion=None, logging_meters=None, store=None):

    model.eval()
    stats_dict = dict()

    with torch.no_grad():
        print("Evaluating epoch {0}/{1} on val data".format(epoch + 1, args.epochs))
        with tqdm(val_iter) as V:
            for batch in V:

                logits = model(batch) # seq length x batch_size x vocab
                target = batch.target # seq length x batch_size
                l = logits[:target.size()[0], :].permute(0, 2, 1)
                predictions = logits[:target.size()[0], :].argmax(2)

                batch_loss = criterion(l, target)

                for name, meter in logging_meters.items():
                    if name == 'loss':
                        meter.update(batch_loss)
                    else:
                        meter.process_batch(predictions, target, model)

            for name, meter in logging_meters.items():
                stats_dict[name] = meter.result()

        if store is not None:
            store["logs"].append_row(stats_dict)

    return stats_dict

def train(model, train_iterator, validation_iter, logging_meters, store, args, ignore_index=None):

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    early_stopping = EarlyStopping(patience = args.patience, verbose = False,
        filename = os.path.join(store.path, CKPT_NAME_LATEST), delta=0.005)
    
    for epoch in range(args.epochs):
    
        new_meters = dict()
        new_meters['sentence-level-accuracy'] = SentenceLevelAccuracy()
        new_meters['token-level-accuracy'] = TokenLevelAccuracy()
        new_meters['length-accuracy'] = LengthLevelAccuracy()
        new_meters['loss'] = AverageMetric()

        model.train()
        print("Training epoch {0}/{1} on train data".format(epoch + 1, args.epochs))
        with tqdm(train_iterator) as T:
            for batch in T:
                optimizer.zero_grad()

                decoder_outputs = model(batch)
                
                # TODO: double check this
                pred = decoder_outputs.permute(1, 2, 0)
                target = batch.target.permute(1, 0)
                batch_loss = criterion(pred, target)

                batch_loss.backward()
                optimizer.step()

                new_meters['loss'].update(batch_loss.item())
                T.set_postfix(loss=new_meters['loss'].result())

        eval_stats = evaluate(model, validation_iter, epoch, args, criterion, logging_meters=new_meters, store=store)

        for name, stat in eval_stats.items():
            if 'accuracy' in name:
                stat = stat * 100
            print('{:<25s} {:.5} {:s}'.format(name, stat, '%' if 'accuracy' in name else ''))

        early_stopping(eval_stats['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping. Loading model from last saved checkoint.")
            model.load_state_dict(torch.load(os.path.join(store.path, CKPT_NAME_LATEST)))
            break

        torch.save(model.state_dict(), os.path.join(store.path, CKPT_NAME_LATEST))

