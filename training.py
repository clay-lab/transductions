import torch
import torchtext.data as tt
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import numpy as np
import os
from early_stopping import EarlyStopping
import seq2seq as ss
from typing import Dict
import cox.store as cx
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

CKPT_NAME_LATEST = 'checkpoint.pt'
		
def test(model: ss.Seq2Seq, test_iter: tt.Iterator, task: str, filename: str):

	model.eval()

	if not os.path.exists('results'):
		os.makedirs('results')
		
	outfile = os.path.join('results', filename + '.tsv')
	with open(outfile, 'w') as f:
		f.write('{0}\t{1}\t{2}\n'.format('source', 'target', 'prediction'))
	with torch.no_grad():
		print("Testing on test data")
		# for batch in test_iter:
		with tqdm(test_iter) as t:
			for batch in t:

				logits = model(batch)
				target = batch.target 
				# predictions = logits[:target.size()[0], :].argmax(2)
				predictions = logits.argmax(2)
				sentences = model.scores2sentence(batch.source, model.encoder.vocab)
				# predictions = model.scores2sentence(predictions, model.decoder.vocab)
				predictions = model.scores2sentence(logits.argmax(2), model.decoder.vocab)
				target = model.scores2sentence(target, model.decoder.vocab)

				with open(outfile, 'a') as f:
					for i, _ in enumerate(sentences):
						f.write('{0}\t{1}\t{2}\n'.format(
							sentences[i], target[i], predictions[i])
						)

def evaluate(model: ss.Seq2Seq, val_iter: tt.Iterator, epoch: int, 
		         args: Dict, criterion=None, logging_meters=None, store=None):

	model.eval()
	stats_dict = dict()

	with torch.no_grad():
		print("Evaluating epoch {0}/{1} on val data".format(epoch + 1, args.epochs))
		with tqdm(val_iter) as V:
			for batch in V:
			# for batch in val_iter:
				logits = model(batch) # seq length x batch_size x vocab
				target = batch.target # seq length x batch size

				perm_logits = logits.permute(1, 2, 0)
				perm_target = batch.target.permute(1, 0) # seq length x batch_size

				pad_len = perm_logits.size()[2] - perm_target.size()[1]
				pad_token = model.decoder.vocab['<pad>']
				new_target = F.pad(perm_target, (0, pad_len) , "constant", pad_token)

				batch_loss = criterion(perm_logits, new_target)
			
			for name, meter in logging_meters.items():
				if name == 'loss':
					meter.update(batch_loss)
				else:
					#meter.process_batch(logits[:target.size()[0], :].argmax(2), target, model)
					meter.process_batch(logits[:target.size()[0], :].argmax(2), target[:logits.size()[0]], model)
		for name, meter in logging_meters.items():
			stats_dict[name] = meter.result()
			meter.reset()

		if store is not None:
			store["logs"].append_row(stats_dict)

	return stats_dict

def train(model: ss.Seq2Seq, train_iterator: tt.Iterator, 
	        validation_iter: tt.Iterator, logging_meters: Dict, 
	        store: cx.Store, args: Dict, save_dir: str, ignore_index=None):

	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index)
	

	model_path = os.path.join(save_dir, 'model.pt')
	checkpoint_path = os.path.join(save_dir, 'checkpoint.pt')

	early_stopping = EarlyStopping(patience = args.patience, verbose = False,
		filename = checkpoint_path, delta=0.005)
	  
	for epoch in range(args.epochs):

		model.train()
		print("Training epoch {0}/{1} on train data".format(epoch + 1, args.epochs))
		with tqdm(train_iterator) as T:
			for batch in T:
			# for batch in train_iterator:
				optimizer.zero_grad()

				decoder_outputs = model(batch)
				pred = decoder_outputs.permute(1, 2, 0)
				target = batch.target.permute(1, 0)

				batch_loss = criterion(pred, target)

				batch_loss.backward()
				optimizer.step()

				logging_meters['loss'].update(batch_loss.item())
				# T.set_postfix(loss=logging_meters['loss'].result())

		eval_stats = evaluate(model, validation_iter, epoch, args, criterion,
		                      logging_meters=logging_meters, store=store)

		for name, stat in eval_stats.items():
			if 'accuracy' in name:
				stat = stat * 100
			sign = '%' if 'accuracy' in name else ''
			print('{:<25s} {:.5} {:s}'.format(name, stat, sign))

		early_stopping(eval_stats['loss'], model)
		if early_stopping.early_stop:
			print("Early stopping, resetting to last checkpoint.")
			model.load_state_dict(torch.load(checkpoint_path))
			break


		# Save the paramaters at the end of every epoch
		torch.save(model.state_dict(), checkpoint_path)

	# Save the entire model so that we can load it in testing without knowledge
	# of the model structure
	torch.save(model, model_path)

