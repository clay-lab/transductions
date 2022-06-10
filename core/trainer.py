import logging
import os
import pickle
import random
import re
from cmd import Cmd
from typing import Dict, List, Tuple

import hydra
import numpy as np
import omegaconf
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader
from torchtext.data import Batch
from torchtext.vocab import Vocab
from tqdm import tqdm

from core.dataset.base_dataset import TransductionDataset
from core.dataset.tpdn_dataset import TPDNDataset
from core.early_stopping import EarlyStopping
from core.metrics.base_metric import (
    LengthAccuracy,
    LossMetric,
    NthTokenAccuracy,
    SequenceAccuracy,
    TokenAccuracy,
)
from core.metrics.meter import Meter
from core.models.base_model import TransductionModel
from core.models.model_io import ModelIO
from core.models.tpn_model import TensorProductEncoder

log = logging.getLogger(__name__)


class Trainer:
    """
    Handles interface between:
      - TransductionModel
      - TransductionDataset
      - Checkpoint
      - Meter
    """

    def __init__(self, cfg: DictConfig):

        self._cfg = cfg
        self._instantiate()

    def _instantiate(self):

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("DEVICE: {}".format(self._device))

        # Check if a random seed should be set
        if "seed" in self._cfg.experiment.hyperparameters.keys():
            random_seed = int(self._cfg.experiment.hyperparameters.seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed(random_seed)
            torch.backends.cudnn.deterministic = True

    def _normalize_lengths(
        self, output: Tensor, target: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Pad the output or target with decoder's <pad> so that loss can be computed.
        """

        diff = int(output.shape[-1] - target.shape[-1])
        pad_idx = self._model._decoder.vocab.stoi["<pad>"]

        if diff == 0:
            pass
        elif diff > 0:
            # output is longer than target. Add <pad> index to end
            # of target until length is equal
            target = F.pad(input=target, pad=(0, diff), value=pad_idx)
        else:
            # target is longer than output. Add one-hot tensor hot in
            # index == pad_idx to end until length is equal
            padding = torch.zeros(output.shape[0], output.shape[1], -diff).to(
                self._device
            )
            padding[:, pad_idx] = 1.0
            output = torch.cat((output, padding), dim=2)

        return output.to(self._device), target.to(self._device)

    def _load_model(
        self, cfg: DictConfig, src_vocab: Vocab, tgt_vocab: Vocab, from_path: str = None
    ):

        model = TransductionModel(cfg, src_vocab, tgt_vocab, self._device)
        log.info(model)

        if from_path is not None:
            log.info("Loading model weights from {}".format(from_path))
            model.load_state_dict(torch.load(from_path))

        return model

    def _load_dataset(self, cfg: DictConfig, from_paths: Dict = None):

        BERT = cfg.model.encoder.unit == "BERT"

        if from_paths is not None:
            src_field = (
                pickle.load(open(from_paths["source"], "rb")) if not BERT else None
            )
            tgt_field = pickle.load(open(from_paths["target"], "rb"))
            dataset = TransductionDataset(
                cfg, self._device, {"source": src_field, "target": tgt_field}, BERT=BERT
            )
        else:
            dataset = TransductionDataset(cfg, self._device, BERT=BERT)

        log.info(dataset)
        return dataset

    def _load_checkpoint(self, from_path: str = None):
        """
        Sets up self._model and self._dataset. If given a path, it will attempt to load
        these from disk; otherwise, it will create them from scratch.
        """

        if from_path is not None:
            chkpt_dir = hydra.utils.to_absolute_path(from_path)
            field_paths = {
                "source": os.path.join(chkpt_dir, "source.pt"),
                "target": os.path.join(chkpt_dir, "target.pt"),
            }
            model_path = os.path.join(chkpt_dir, "model.pt")
            self._dataset = self._load_dataset(self._cfg.experiment, field_paths)
        else:
            model_path = None
            self._dataset = self._load_dataset(self._cfg.experiment)

        src_field = self._dataset.source_field
        tgt_field = self._dataset.target_field
        src_vocab = src_field.vocab if hasattr(src_field, "vocab") else None
        self._model = self._load_model(
            self._cfg.experiment, src_vocab, tgt_field.vocab, model_path
        )

    def train(self):

        # Load checkpoint
        self._load_checkpoint()

        log.info("Beginning training")

        lr = self._cfg.experiment.hyperparameters.lr
        tf_ratio = self._cfg.experiment.hyperparameters.tf_ratio
        if not tf_ratio:
            # TODO: Should probably come up with better logic to handle the 'null' case
            tf_ratio = 0.0
        epochs = self._cfg.experiment.hyperparameters.epochs
        optimizer = torch.optim.SGD(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(
            weight=None, ignore_index=self._model._decoder.PAD_IDX
        )

        early_stoping = EarlyStopping(self._cfg.experiment.hyperparameters)

        pad_idx = self._model._decoder.vocab.stoi["<pad>"]

        # Metrics
        seq_acc = SequenceAccuracy(pad_idx)
        tok_acc = TokenAccuracy(pad_idx)
        len_acc = LengthAccuracy(pad_idx)
        first_acc = NthTokenAccuracy(n=1)
        avg_loss = LossMetric(lambda p, t: F.cross_entropy(p, t, ignore_index=pad_idx))

        meter = Meter([seq_acc, tok_acc, len_acc, first_acc, avg_loss])

        for epoch in range(epochs):

            log.info("EPOCH %i / %i", epoch + 1, epochs)

            log.info("Computing metrics for 'train' dataset")
            self._model.train()

            with tqdm(self._dataset.iterators["train"]) as T:
                for batch in T:

                    optimizer.zero_grad()

                    # Loss expects:
                    #   output:  [batch_size, classes, seq_len]
                    #   target: [batch_size, seq_len]
                    output = self._model(batch, tf_ratio=tf_ratio).permute(1, 2, 0)
                    target = batch.target.permute(1, 0)
                    output, target = self._normalize_lengths(output, target)

                    # TODO: Loss should ignore <pad>, ...maybe others?
                    loss = criterion(output, target)

                    loss.backward()
                    optimizer.step()

                    # Compute metrics
                    meter(output, target)

                    T.set_postfix(trn_loss="{:4.3f}".format(loss.item()))

                meter.log(stage="train", step=epoch)
                meter.reset()

            # Perform val, test, gen, ... passes
            with torch.no_grad():

                log.info("Computing metrics for 'val' dataset")
                with tqdm(self._dataset.iterators["val"]) as V:
                    val_loss = 0.0
                    for batch in V:

                        output = self._model(batch).permute(1, 2, 0)
                        target = batch.target.permute(1, 0)
                        output, target = self._normalize_lengths(output, target)

                        meter(output, target)

                        # Compute average validation loss
                        val_loss = F.cross_entropy(output, target, ignore_index=pad_idx)
                        V.set_postfix(val_loss="{:4.3f}".format(val_loss.item()))

                    meter.log(stage="val", step=epoch)
                    meter.reset()

                log.info("Computing metrics for 'test' dataset")
                with tqdm(self._dataset.iterators["test"]) as T:
                    for batch in T:

                        output = self._model(batch).permute(1, 2, 0)
                        target = batch.target.permute(1, 0)
                        output, target = self._normalize_lengths(output, target)

                        meter(output, target)

                    meter.log(stage="test", step=epoch)
                    meter.reset()

                if "gen" in list(self._dataset.iterators.keys()):
                    log.info("Computing metrics for 'gen' dataset")
                    with tqdm(self._dataset.iterators["gen"]) as G:
                        for batch in G:

                            output = self._model(batch).permute(1, 2, 0)
                            target = batch.target.permute(1, 0)
                            output, target = self._normalize_lengths(output, target)

                            meter(output, target)

                        meter.log(stage="gen", step=epoch)
                        meter.reset()

                other_iters = list(self._dataset.iterators.keys())
                other_iters = [
                    o for o in other_iters if o not in ["train", "val", "test", "gen"]
                ]
                for itr in other_iters:
                    log.info("Computing metrics for '{}' dataset".format(itr))
                    with tqdm(self._dataset.iterators[itr]) as T:
                        for batch in T:

                            output = self._model(batch).permute(1, 2, 0)
                            target = batch.target.permute(1, 0)
                            output, target = self._normalize_lengths(output, target)

                            meter(output, target)

                        meter.log(stage=itr, step=epoch)
                        meter.reset()

            should_stop, should_save = early_stoping(val_loss)
            if should_save:
                torch.save(self._model.state_dict(), "model.pt")
            if should_stop:
                break

    def eval(self, eval_cfg: DictConfig):

        # Load checkpoint data
        self._load_checkpoint(eval_cfg.checkpoint_dir)

        pad_idx = self._dataset.target_field.vocab.stoi["<pad>"]

        # Create meter
        seq_acc = SequenceAccuracy(pad_idx)
        tok_acc = TokenAccuracy(pad_idx)
        len_acc = LengthAccuracy(pad_idx)
        first_acc = NthTokenAccuracy(n=1)
        avg_loss = LossMetric(lambda p, t: F.cross_entropy(p, t, ignore_index=pad_idx))

        meter = Meter([seq_acc, tok_acc, len_acc, first_acc, avg_loss])

        log.info("Beginning evaluation")
        self._model.eval()

        for key in list(self._dataset.iterators.keys()):

            log.info("Evaluating model on {} dataset".format(key))

            with open("{}.tsv".format(key), "a") as f:
                f.write("source\ttarget\tprediction\n")

                with tqdm(self._dataset.iterators[key]) as T:
                    for batch in T:

                        source = batch.source.permute(1, 0)
                        prediction = self._model(batch).permute(1, 2, 0)
                        target = batch.target.permute(1, 0)
                        prediction, target = self._normalize_lengths(prediction, target)

                        # TODO: SHOULD WE USE normed ouputs instead of prediction ehre?
                        meter(prediction, target)

                        src_toks = self._model._encoder.to_tokens(source)
                        pred_toks = self._model._decoder.to_tokens(prediction.argmax(1))
                        tgt_toks = self._model._decoder.to_tokens(target)

                        for seq in range(len(tgt_toks)):
                            src_line = " ".join(src_toks[seq])
                            tgt_line = " ".join(tgt_toks[seq])
                            pred_line = " ".join(pred_toks[seq])

                            f.write(
                                "{}\t{}\t{}\n".format(src_line, tgt_line, pred_line)
                            )

                meter.log(stage=key, step=0)
                meter.reset()

    def arith_eval(self, eval_cfg: DictConfig):

        # Load checkpoint data
        self._load_checkpoint(eval_cfg.checkpoint_dir)

        pad_idx = self._dataset.target_field.vocab.stoi["<pad>"]

        # Create meter
        seq_acc = SequenceAccuracy(pad_idx)
        len_acc = LengthAccuracy(pad_idx)
        object_acc = NthTokenAccuracy(n=5)

        meter = Meter([seq_acc, len_acc, object_acc])

        # Create evaluation dataset
        with omegaconf.open_dict(eval_cfg):
            eval_cfg.hyperparameters = self._cfg.experiment.hyperparameters
            # eval_cfg.hyperparameters.batch_size=3

        dataset = TransductionDataset(
            eval_cfg,
            self._device,
            fields={
                "source": self._dataset.source_field,
                "target": self._dataset.target_field,
            },
        )

        log.info("Beginning evaluation")
        self._model.eval()

        for key in list(dataset.iterators.keys()):

            log.info("Evaluating model on {} dataset".format(key))

            with open("{}.tsv".format(key), "a") as f:
                f.write("source\ttarget\tprediction\n")

                with tqdm(dataset.iterators[key]) as T:
                    for batch in T:

                        # Perform arithmetic computation by reducing an input batch
                        source = batch.source.permute(1, 0)
                        prediction = self._model.forward_batch_expr(
                            batch, offset=eval_cfg.dataset.offset
                        ).permute(1, 2, 0)
                        target = batch.target.permute(1, 0)
                        prediction, target = self._normalize_lengths(prediction, target)

                        # TODO: SHOULD WE USE normed ouputs instead of prediction ehre?
                        meter(prediction, target)

                        src_toks = self._model._encoder.to_tokens(source)
                        pred_toks = self._model._decoder.to_tokens(prediction.argmax(1))
                        tgt_toks = self._model._decoder.to_tokens(target)

                        # print(src_toks)

                        for seq in range(len(tgt_toks)):
                            src_line = " ".join(src_toks[seq])
                            tgt_line = " ".join(tgt_toks[seq])
                            pred_line = " ".join(pred_toks[seq])

                            f.write(
                                "{}\t{}\t{}\n".format(src_line, tgt_line, pred_line)
                            )

                meter.log(stage=key, step=0)
                meter.reset()

    def _load_tpdn_data(self, cfg: DictConfig):

        self._get_encodings(
            splits=cfg.splits, use_cached=cfg.use_cached, role_scheme=cfg.role_scheme
        )
        iterators = {}
        for split in cfg.splits:
            iterators[split] = DataLoader(
                TPDNDataset(f"{split}-{cfg.role_scheme}-enc.pkl"),
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle_data,
                collate_fn=TPDNDataset.collate_fn,
            )

        return iterators

    def _roles_for_entry(self, source: np.ndarray, role_scheme: str):

        fillers = source
        for tok in [
            self._dataset.source_field.vocab.stoi["<sos>"],
            self._dataset.source_field.vocab.stoi["<eos>"],
            self._dataset.source_field.vocab.stoi["<pad>"],
            self._dataset.source_field.vocab.stoi["<unk>"],
        ]:
            fillers = fillers[fillers != tok]

        if role_scheme == "ltr":
            roles = np.array([p for p in range(fillers.shape[0])])
        elif role_scheme == "rtl":
            roles = np.array(reversed([p for p in range(fillers.shape[0])]))
        elif role_scheme == "bow":
            roles = np.array([0 for _ in range(fillers.shape[0])])
        elif role_scheme == "refsub":
            # Reflexive tokens get treated as subjects
            roles = [p for p in range(fillers.shape[0])]
            ref_index = np.where(fillers == 38, 1, 0)
            for i, v in enumerate(ref_index):
                if v == 1:
                    roles[i] = 1
            roles = np.array(roles)
        elif role_scheme == "s-v-or":
            roles = np.array([p for p in range(fillers.shape[0])])
        elif role_scheme == "svot":
            roles = np.array([p for p in range(fillers.shape[0])] + [3])
        else:
            raise ValueError

        return roles

    def _fillers_for_entry(self, source: np.ndarray, role_scheme: str):

        if role_scheme == "ltr" or role_scheme == "rtl":
            fillers = source
            for tok in [
                self._dataset.source_field.vocab.stoi["<sos>"],
                self._dataset.source_field.vocab.stoi["<eos>"],
                self._dataset.source_field.vocab.stoi["<pad>"],
                self._dataset.source_field.vocab.stoi["<unk>"],
            ]:
                fillers = fillers[fillers != tok]

        elif role_scheme == "s-v-or":
            fillers = source
            for tok in [
                self._dataset.source_field.vocab.stoi["<sos>"],
                self._dataset.source_field.vocab.stoi["<eos>"],
                self._dataset.source_field.vocab.stoi["<pad>"],
                self._dataset.source_field.vocab.stoi["<unk>"],
            ]:
                fillers = fillers[fillers != tok]

            ref_toks = [
                self._dataset.source_field.vocab.stoi["himself"],
                self._dataset.source_field.vocab.stoi["herself"],
            ]
            fillers[fillers == ref_toks] = ref_toks[0]

        elif role_scheme == "bow":
            fillers = source
            for tok in [
                self._dataset.source_field.vocab.stoi["<sos>"],
                self._dataset.source_field.vocab.stoi["<eos>"],
                self._dataset.source_field.vocab.stoi["<pad>"],
                self._dataset.source_field.vocab.stoi["<unk>"],
            ]:
                fillers = fillers[fillers != tok]
        elif role_scheme == "svot":
            fillers = source
            for tok in [
                self._dataset.source_field.vocab.stoi["<sos>"],
                self._dataset.source_field.vocab.stoi["<eos>"],
                self._dataset.source_field.vocab.stoi["<pad>"],
                self._dataset.source_field.vocab.stoi["<unk>"],
            ]:
                fillers = fillers[fillers != tok]

            ref_toks = [
                self._dataset.source_field.vocab.stoi["himself"],
                self._dataset.source_field.vocab.stoi["herself"],
            ]
            if np.isin(ref_toks, fillers).any():
                clause_type = 1
            else:
                clause_type = 2
            fillers = np.append(fillers, [clause_type])

        else:
            raise NotImplementedError

        return fillers

    def _get_encodings(self, splits: List[str], use_cached=True, role_scheme="ltr"):

        for split in splits:
            fname = f"{split}-{role_scheme}-enc.pkl"
            if os.path.isfile(fname) and use_cached:
                log.info(f"Found {split} encodings at {fname}")
                pass
            else:
                log.info(f"Generating encodings for {split} set")
                df = pd.DataFrame(
                    columns=["fillers", "roles", "annotation", "encoding", "target"]
                )
                with tqdm(self._dataset.iterators[split]) as T:
                    for batch in T:
                        enc = self._model._encoder(batch)
                        for k in range(batch.source.shape[1]):
                            source = batch.source[:, k].detach().numpy()
                            roles = self._roles_for_entry(source, role_scheme)
                            fillers = self._fillers_for_entry(source, role_scheme)
                            encodings = enc.enc_outputs[:, k, :].detach().numpy()
                            annotation = batch.annotation[:, k].detach().numpy()
                            target = batch.target[:, k].detach().numpy()
                            df = df.append(
                                {
                                    "roles": roles,
                                    "fillers": fillers,
                                    "annotation": annotation,
                                    "encoding": encodings,
                                    "target": target,
                                },
                                ignore_index=True,
                            )
                df.to_pickle(fname)

    def fit_tpdn(self, tpdn_cfg: DictConfig):

        # Load checkpoint data
        self._load_checkpoint(tpdn_cfg.checkpoint_dir)
        log.info(f"Fitting TPDN to checkpoint at {tpdn_cfg.checkpoint_dir}")

        self._model.eval()

        tpdn_iterators = self._load_tpdn_data(tpdn_cfg.data)

        if tpdn_cfg.model.num_fillers is None:
            tpdn_cfg.model.num_fillers = self._model._encoder.vocab_size
        tpdn = TensorProductEncoder(tpdn_cfg, self._device)
        log.info(tpdn)

        # Check if TPDN model exists
        if not (
            os.path.isfile(f"tpdn-{tpdn_cfg.data.role_scheme}.pt")
            and tpdn_cfg.model.use_cached
        ):

            # Train TPDN
            optimizer = torch.optim.Adam(
                tpdn.parameters(), lr=tpdn_cfg.hyperparameters.lr
            )
            criterion = nn.MSELoss()
            epochs = tpdn_cfg.hyperparameters.epochs

            log.info("Training TPDN model")
            tpdn.train()
            for epoch in range(epochs):

                log.info("EPOCH %i / %i", epoch + 1, epochs)

                with tqdm(tpdn_iterators["train"]) as T:
                    for batch in T:
                        optimizer.zero_grad()

                        fillers, roles, _, encodings, _ = batch

                        target = encodings[:, -1, :].unsqueeze(0)
                        input = ModelIO({"fillers": fillers, "roles": roles})
                        output = tpdn(input)

                        loss = criterion(output.bound_embeddings, target)

                        loss.backward()
                        optimizer.step()

                        T.set_postfix(trn_loss="{:4.3f}".format(loss.item()))

            torch.save(tpdn.state_dict(), f"tpdn-{tpdn_cfg.data.role_scheme}.pt")

        fpath = os.path.join(os.getcwd(), f"tpdn-{tpdn_cfg.data.role_scheme}.pt")
        log.info(f"Reading TPDN model from {fpath}")
        tpdn.load_state_dict(torch.load(fpath))

        # Compute substitution accuracy
        tpdn.eval()
        disp_loss = nn.CrossEntropyLoss()

        pad_idx = self._dataset.target_field.vocab.stoi["<pad>"]
        meter = Meter(
            [
                SequenceAccuracy(pad_idx),
                TokenAccuracy(pad_idx),
                LengthAccuracy(pad_idx),
                NthTokenAccuracy(n=1),
                NthTokenAccuracy(n=3),
                NthTokenAccuracy(n=5),
            ]
        )

        log.info("Beginning evaluation")
        self._model.eval()

        for key in ["train", "val", "test", "gen"]:

            log.info("Evaluating model on {} dataset".format(key))
            with open(f"{key}-{tpdn_cfg.data.role_scheme}-pred.tsv", "w") as f:

                with tqdm(tpdn_iterators[key]) as T:
                    for batch in T:

                        fillers, roles, annotation, _, targets = batch
                        annotation = annotation.permute(1, 0)
                        # encodings = torch.reshape(
                        #     encodings, (roles.shape[0], roles.shape[1], -1)
                        # )
                        input = ModelIO({"fillers": fillers, "roles": roles})

                        tpdn_encoding = tpdn(input)
                        tpdn_encoding.set_attributes({"transform": annotation})
                        tpdn_encoding.set_attribute(
                            "enc_outputs", tpdn_encoding.bound_embeddings
                        )

                        hybrid_outputs = self._model._decoder(
                            tpdn_encoding, tf_ratio=0.0
                        ).dec_outputs
                        prediction = hybrid_outputs.permute(1, 2, 0)
                        prediction, target = self._normalize_lengths(
                            prediction, targets
                        )

                        src_toks = self._model._encoder.to_tokens(fillers)
                        pred_toks = self._model._decoder.to_tokens(prediction.argmax(1))
                        tgt_toks = self._model._decoder.to_tokens(target)

                        for seq in range(len(tgt_toks)):
                            src_line = " ".join(src_toks[seq])
                            tgt_line = " ".join(tgt_toks[seq])
                            pred_line = " ".join(pred_toks[seq])

                            if tgt_line != pred_line:
                                score = 0
                            else:
                                score = 1

                            f.write(
                                "{}\t{}\t{}\t{}\n".format(
                                    src_line, tgt_line, pred_line, score
                                )
                            )

                        loss = disp_loss(prediction, target)
                        meter(prediction, target)
                        T.set_postfix(loss="{:4.3f}".format(loss.item()))

                meter.log(stage=key, step=0)
                meter.reset()

    def repl(self, repl_cfg: DictConfig):

        # Load checkpoint data
        self._load_checkpoint(repl_cfg.checkpoint_dir)

        log.info("Beginning REPL")

        repl = ModelREPL(self._model, self._dataset)
        repl.cmdloop()

    def arith(self, repl_cfg: DictConfig):

        # Load checkpoint data
        self._load_checkpoint(repl_cfg.checkpoint_dir)

        log.info("Beginning REPL")

        repl = ModelArithmeticREPL(self._model, self._dataset)
        repl.cmdloop()

    def get_trajectories(self, input: str):

        repl = ModelREPL(model=self._model, dataset=self._dataset)
        batch = repl.batchify(input)

        enc_input = ModelIO({"source": batch.source})
        enc_output = self._model._encoder(enc_input)

        enc_output.set_attributes(
            {"source": batch.source, "transform": batch.annotation}
        )

        if hasattr(batch, "target"):
            enc_output.set_attribute("target", batch.target)

        dec_output = self._model._decoder(enc_output, tf_ratio=0.0)

        return enc_output, dec_output

    def plot_trajectories(self, input: str):

        repl = ModelREPL(model=self._model, dataset=self._dataset)
        batch = repl.batchify(input)

        enc_input = ModelIO({"source": batch.source})
        diffs = {}
        space = np.array([np.linspace(-1.0, 1.0, num=5) for _ in range(250)])
        for s in space:
            h = torch.tensor(s, dtype=torch.float)
            enc_output = self._model._encoder.forward_with_hidden(enc_input, hidden=h)
            delta = (enc_output.enc_hidden - h) / 10.0
            h_idx = tuple(h.flatten().detach().numpy())
            diffs[h_idx] = delta

        return diffs


class ModelREPL(Cmd):
    """
    A REPL for interacting with a pre-trained model.
    """

    prompt = "> "

    def __init__(self, model: TransductionModel, dataset: TransductionDataset):
        super(ModelREPL, self).__init__()
        self.intro = "Enter sequences into the model for evaluation."
        self._model = model
        self._dataset = dataset

    def batchify(self, args):
        """
        Turn the REPL input into a batch for the model to process.
        """

        transf, source = args.split(" ", 1)

        source = source.split(" ")
        source.append("<eos>")
        source.insert(0, "<sos>")
        source = [self._model._encoder.to_ids([s]) for s in source]
        source = torch.LongTensor(source)

        transf = ["<sos>", transf, "<eos>"]
        transf = [[self._dataset.transform_field.vocab.stoi[t]] for t in transf]
        transf = torch.LongTensor(transf)

        batch = Batch()
        batch.source = source
        batch.annotation = transf

        return batch

    def default(self, args):

        batch = self.batchify(args)

        prediction = (
            self._model(batch, plot_trajectories=True).permute(1, 2, 0).argmax(1)
        )
        prediction = self._model._decoder.to_tokens(prediction)[0]
        prediction = " ".join(prediction)

        source = self._model._encoder.to_tokens(batch.source.permute(1, 0))[0]
        source = " ".join(source)

        transformation = self._model._decoder.to_tokens(batch.annotation.permute(1, 0))[
            0
        ]
        transformation = " ".join(transformation)

        result = "{} → {} → {}".format(source, transformation, prediction)
        log.info(result)

    def do_quit(self, args):
        log.info("Exiting REPL.")
        raise SystemExit


class ModelArithmeticREPL(ModelREPL):
    """
    A REPL for doing expression math
    """

    prompt = "λ "

    def default(self, args):

        # Extract transform from expressions
        transform = args.split()[0]
        expressions = args[len(transform) :]

        # Split expressions on + and - operators
        expressions = re.split(r"(\+|\-)", expressions)
        expressions = list(filter(None, [e.strip() for e in expressions]))

        batched_expressions = []

        for e in expressions:
            if e in "+-":
                batched_expressions.append(e)
            else:
                batch = self.batchify(f"{transform} {e}")
                batched_expressions.append(batch)

        prediction = (
            self._model.forward_expression(batched_expressions)
            .permute(1, 2, 0)
            .argmax(1)
        )
        prediction = self._model._decoder.to_tokens(prediction)[0]
        prediction = " ".join(prediction)

        source = " ".join(args.split(" ")[1:])

        result = "{} = {}".format(source, prediction)
        log.info(result)

    def do_eos(self, args):

        expr_list = re.split("+|-", args)
        expr_list = list(filter(None, [e.strip() for e in args]))

        print(expr_list)

        transform = expr_list[0]
        unbatched_expressions = expr_list[1:]

        expressions = []

        for e in unbatched_expressions:
            if e in "+-":
                expressions.append(e)
            else:
                batch = self.batchify(f"{transform} {e}")
                expressions.append(batch)

        prediction = (
            self._model.forward_expression_eos(expressions).permute(1, 2, 0).argmax(1)
        )
        prediction = self._model._decoder.to_tokens(prediction)[0]
        prediction = " ".join(prediction)

        source = " ".join(args.split(" ")[1:])

        result = "{} = {}".format(source, prediction)
        log.info(result)

    def do_quit(self, args):
        log.info("Exiting REPL.")
        raise SystemExit
