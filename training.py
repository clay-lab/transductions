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

def evaluate(model: ss.Seq2Seq, val_iter: tt.Iterator, epoch: int, 
		         args: Dict, criterion=None, logging_meters=None, store=None):

	model.eval()
	stats_dict = dict()

	with torch.no_grad():
		print("Evaluating epoch {0}/{1} on val data".format(epoch + 1, args.epochs))
		with tqdm(val_iter) as V:
			for batch in V:
				logits = model(batch) # seq length x batch_size x vocab
				target = batch.target # seq length x batch size

				logits_max = logits.argmax(2) # seq length x batch_size
				# pad sequence combines them into a single tensor and pads whichever is shorter
				# then we split along that new dimension to recover separate prediction and target tensors
				pad_token = model.decoder.vocab['<pad>']
				padded_combined = pad_sequence([logits_max, batch.target], padding_value=pad_token)
				prediction_padded, target_padded = padded_combined[:, 0, :], padded_combined[:, 1, :]				
			
				# This block pads first dimension of logits_padded to be target_padded.shape[0]
				logits_padded = torch.Tensor(target_padded.shape[0], logits.shape[1], logits.shape[2])
				logits_padded.fill_(pad_token)
				logits_padded[:logits.shape[0], :, :] = logits
				# logits_padded = logits # THIS IS WRONG!!!! sometimes logits might have to be padded up to the length of target_padded

				perm_logits = logits_padded.permute(1, 2, 0) # batch_size x vocab x seq length
				target_padded_perm = target_padded.permute(1, 0) # seq length x batch_size
				batch_loss = criterion(perm_logits, target_padded_perm)

			for name, meter in logging_meters.items():
				if name == 'loss':
					meter.update(batch_loss)
				else:
					meter.process_batch(prediction_padded, target_padded, model)
		for name, meter in logging_meters.items():
			stats_dict[name] = meter.result()
			meter.reset()

		if store is not None:
			store["logs"].append_row(stats_dict)

	return stats_dict

def train(model: ss.Seq2Seq, train_iterator: tt.Iterator, 
	        validation_iter: tt.Iterator, logging_meters: Dict, 
	        store: cx.Store, args: Dict, save_dir: str, ignore_index=None):

	optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
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
				T.set_postfix(loss=logging_meters['loss'].result())

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

