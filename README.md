# Transductions

A Pytorch/Torchtext framework for building and testing Sequence-to-Sequence
models and experiments. Transductions relies heavily on 
[Hydra](https://github.com/facebookresearch/hydra) for configuration,
allowing you to specify model architectures, datasets, and experiments using
YAML files for easy modularity and reproducibility. Out of the box, Transductions
supports the [Tensorboard](https://tensorboard.dev) logging framework to make
recording model performance easy.

Transductions currently supports the following model architectures:
- **Encoders**
  * SRN
  * GRU
  * LSTM
  * Transformer
- **Decoders**
  * SRN
  * GRU
  * LSTM
  * Transformer

Recurrent currently support `null`, `Additive`, `Multiplicative`, and `DotProduct` attention;
Transformers of course implement multi-head attention.

## Installation

Transductions uses `poetry` to manage dependencies, making installation easy. Once you have cloned the repository,
run `poetry shell && poetry install` to activate a virtual environment and install all necessary dependencies.

## Training

Models are trained via the `train.py` script, which is in turn configured using
the `conf/train.yaml` file, shown below:
```YAML
defaults:
  - experiment: ???
  - hydra/output: custom

pretty_print: true
```
The `- experiment: ???` line means that an experiment must be specified on the
command line. To try it out, run the following command:
```bash
python train.py experiment=test
```
This loads up the example experiment defined in `conf/experiment/test.yaml` and
begins training a model. Every experiment configuration file must specify four
things:
 - A `name`, which is a unique identifier in the `outputs/` directory to group
   all runs of the experiment together.
 - A `dataset`, which is the name of a dataset configuration file in the `conf/dataset/`
   directory.
 - A `model`, which is the name of a model configuration file in the `conf/model/` 
   directory.
 - `hyperparameters`, which is the name of a hyperparameter configuration file in the `conf/hyperparameters/`
   directory.

This is the `test.yaml` file:
```YAML
defaults:
  - /dataset: alice-1
  - /hyperparameters: default
  - /model: test-model

name: test-exp
```
If you want to define your own experiments, make a new YAML file and place it in
the `conf/experiment` directory. Transductions uses Hydra to manage these 
configurations---if you're interested in how this works, check out their 
[website](hydra.cc) for documentation and examples.

Outputs from a training session (the model weights, a copy of the model and Hydra 
configurations, etc.) are stored in the `outputs/` directory. If this directory
doesn't exist, one will be created for you. Runs are organized in the following
way, as an example:
```
outputs/
  experiment/
    model/
      YYYY-MM-DD_HH-MM-SS/
        .hydra/
        tensorboard/
        model.pt
        source.pt
        target.pt
        train.log
```

### Hyperparameter Configuration
Hyperparameters are configured by options in the `conf/hyperparameters/` directory.
The default configuration file `default.yaml` is somewhat arbitrary, but provides a
useful example for creating customized configurations.
```YAML
# default.yaml

epochs: 100
batch_size: 32
lr: 0.01
tolerance: 0.001
patience: 100
tf_ratio: 0.5

cuda: 0 # -1 means use CPU
```
This file sets the following options:
- `epochs`: The maximum number of epochs for training. 
- `batch_size`: The batch size used during training and evaluation.
- `lr`: The learning rate used. Currently, `transductions` only supports using an SGD optimizer (from `torch.optim`) with a constant learning rate. Support is planned for alternative optimizers (Adam, AdamW) with adaptive learning rates.
- `tolerance`: How much a model must decrease its validation loss by within `patience` number of epochs to avoid early stopping.
- `patience`: How many epochs a model can run for during training without decreasing validation loss by `tolerance` before early stopping kicks in.
- `tf_ratio` (between `0.0` and `1.0`): The teacher forcing ratio, used during training to determine whether or not a target is used during decoding versus the model's own prediction is used. Higher `tf_ratio` means that the trainer is more likely to use the target, rather than the model's own prediction.
- `cuda`: The GPU Cuda device number to use, or else `-1` if the model should only run on CPU.

**A note on Early Stopping:** Currently, `transductions` implements early stopping for all models, and there isn't yet a good way to turn it off. However, if `patience >= epochs` in the hyperparameter configuration file, this essentially turns off early stopping since the model will always run for `epochs` number of epochs and then stop. It's a bit hacky, and I plan to introduce an actual configuration option for it in the future, but for now this will get the job done.

### Model Configuration
A model configuration specifies details of the model architecture for the encoder and the decoder. Here is an example configuration file which specifies a GRU-Encoder, GRU-Decoder model with no attention.
````YAML
# sequence-gru-inattentive.yaml

name: GRU

encoder:
  unit: GRU
  type: sequence
  dropout: 0
  num_layers: 1
  embedding_size: 256
  hidden_size: 256
  
decoder:
  unit: GRU
  type: sequence
  attention: null
  dropout: 0
  num_layers: 1
  max_length: 30
  embedding_size: 256
  hidden_size: 256
````

The `name` parameter defines a unique name for the model which will be used to identify and group runs of a particular experiment which have been trained using this architecture. For example, a run of `experiment-1` using the `sequence-gru-inattentive.yaml` model would be grouped under `outputs/experiment-1/GRU` since this is the specified `name` of this particular model.

The encoder and decoder have separate configuration options within the file. For both,
- `unit` specifies the type of (recurrent or transformer) unit used in the model. Available options are `GRU`, `SRN`, `LSTM`, or `Transformer`.
- `type` specifies the format of the data to be read in (for the encoder) or produced (by the decoder). Right now, the only valid option is `sequence` (which implements the classic seq-to-seq architecture) but it is planned to also allow for tree/graph based inputs and outputs as well in the future.
- `dropout`: The probability of dropout in the unit.
- `num_layers`: How many layers the encoder or decoder should have for their units.
- `embedding_size`: Size of the embedding dimension.
- `hidden_size`: Size of the hidden layer.

The decoder has two additional properties:
- `attention`: What kind of attention to use. Valid options are `null`, `Additive`, `Multiplicative`, and `DotProduct`.
- `max_length`: The maximum length of sequences that the decoder can produce. Used to prevent the decoder from accidentally producing unbounded sequences and never terminating.

Note that for Transformer models, a slightly different configuration is needed since they implement their own form of requisite self-attention. Here is an example from the `sequence-transformer.yaml` config file:
````YAML
# sequence-transformer.yaml

name: Transformer

encoder:
  unit: Transformer
  type: sequence
  dropout: 0
  num_layers: 1
  hidden_size: 256
  embedding_size: 256
  num_heads: 8
  
decoder:
  unit: Transformer
  type: sequence
  dropout: 0
  num_layers: 1
  max_length: 30
  hidden_size: 256
  embedding_size: 256
  num_heads: 8
  attention: null
````
Note that the `attention` parameter in the decoder is not really used, and so is set to `null`, and both the encoder and decoder have an extra `num_heads` option which configures the number of heads on the multiheaded attention models.

### Dataset Configuration
At a high level, the dataset configuration file specifies the relationship
between an input file, containing the full dataset, and the various splits
which are used in an experiment. By default, Transductions assumes that you
want to generate these splits on the first use of the dataset. To see how this
works, let's walk through the `alice-1.yaml` configuration file, which creates
a dataset used in Frank & Petty, [“Sequence-to-Sequence Networks Learn the 
Meaning of Reflexive Anaphora”](https://www.aclweb.org/anthology/2020.crac-1.16/) (2020):
```YAML
# alice-1.yaml
#
# Withholds reflexive sentences containing "Alice" (e.g., "Alice sees herself")
# during training to explore lexical generalization.

name: alice-1
input: grammar-1.tsv # where is the full dataset
source_format: sequence # 'sequence' or 'tree'
target_format: sequence # 'sequence' or 'tree'
overwrite: False # Always re-create splits from raw data?
transform_field: source # 'source' or 'target', which should include transforms?

splits:
  train: 80
  test: 10
  val: 10

# Defines the generalization set. All inputs which match the provided
# regex will be withheld from the train/test/val splits.
withholding: 
  - '^Alice \w+ herself.*'

# Defines named test sets. For each entry, a .pt file will be created 
# containing all inputs which match the given regex.
tracking:
  alice_subject: '^Alice.*'
  alice_object: '^\w+ \w+ Alice.*'
  alice_reflexive: '^Alice \w+ herself.*'
  alice_subject_transitive: '^Alice \w+ \w+'
  alice_subject_intransitive: '^Alice \w+\t'
  alice_alice: 'Alice \w+ Alice'

  herself: 'herself'
  himself: 'himself'
```

The `name:` parameter specifies a custom identifier used as a directory name
in the `data/processed/` directory. The `input:` parameter is the name of 
the source file containing the full dataset in the `data/raw/` directory.
The `source_format` and `target_format` parameters specify what kind of data
are used for the source and target of the dataset. For the time being, the only
valid choice is `sequence`. The `overwrite` parameter specifies whether or not
the dataset should be re-created every time you kick off a training run. This
should probably be `False` unless you are tweaking the dataset. The `transform_field`
specifies which field, `source` or `target`, should contain the transformation
tokens.

The `splits` parameters (`train`, `test`, `val`) specify how the full dataset
should be randomly split into different splits. Note that the float values 
here must sum to `100`.

The `withholding` parameter specifies a list of strings which are used as 
RegEx matches to withhold a particular entry from the in-distribution splits
and instead place it in a `gen` split.

The `tracking` parameter defines a dictionary of RegExes which are used to
create tracking splits. These are evaluated along side the `train`, `test`,
`val`, and `gen` splits during training every epoch, but they don't affect 
how data are split up or withheld. For each `key : value` combination specified
here, a `key.pt` file will be created in the `data/processed/DATASET/` directory
containing all lines from the input which match the `value` regex.


### Creating Static Datasets
Not every `gen` set or tracking set can be nicely defined by a regular expression.
In this case, it's not possible to dynamically re-created the dataset from the
original source file using Transductions. But, if you manually/externally
separate out the `train/test/val/gen` splits, you can use these as-is without
any of the dynamic processing. To do this, just create a directory in 
`data/processed/` matching the name of your dataset, place the `train.pt`,
`val.pt`, `test.pt`, and `gen.pt` splits there, and make sure that `overwrite:`
is set to `False` in the dataset configuration YAML file.

## Evaluation

There are two ways to evaluate model performance. The first is using `eval.py`, which will load 
a model from the saved weights and evaluate it on every split which was generated for it
during training. 
```bash
python eval.py
```
This will generate an `eval/` directory inside the model's checkpoint directory containing the `eval.log`
log file for the evaluation run along with tab-separate value files for each of the splits in the following
format:
```
source  target  prediction
```

There is also in interactive Read-Evaluate-Print Loop (REPL) which lets you load a model and input arbitrary
transforms sequences to the model, which will print out its predictions. Run
```bash
python repl.py
```
and enter sequences of the form
```
> TRANSFORM this is a sentence
```
where `TRANSFORM` is the transform token. A log for each REPL run will be saved in the `repl/` directory 
inside the model checkpoint directory.

The `eval.py` and `repl.py` scripts are configured by the `eval.yaml` and `repl.yaml` configuration files in 
the `conf/` directory. Both contain a single parameter `checkpoint_dir:` which must be set to point to
a model's checkpoint directory. Just as in the training script, this value may be overridden on the command-line:
```bash
python eval.py checkpoint_dir=FILEPATH
```

## Tensorboard 
Transductions has built-in support for Tensorboard logging during training, allowing you to monitor training
curves and custom metrics. To view a Tensorboard dashboard for a training run, use
```bash
tensorboard --logdir=CHECKPOINT_DIR
```
Tensorboard's real strength is that it *recursively* searches the `logdir` for its log files, which means you
can point it to an arbitrarily high directory and view the logs for multiple runs at once. This is especially
useful for viewing the training curves for an entire experiment, or the curves for a particular model
architecture within an experiment.