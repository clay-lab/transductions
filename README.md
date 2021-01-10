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

Transductions uses `conda` to manage dependencies, making installation easy. Once you have cloned the repository,
create the conda virtual environment by running
```
conda env create --file=.devcontainer/environment.yaml && conda activate transd
```

## Training

To train a new model, run the following script:
```bash
python train.py
```
The `train.py` script is configured through the `train.yaml` file in the `conf`
directory. This YAML file specifies three further configuration options: 
`experiment/hyperparameters`, `experiment/model`, and `experiment/dataset`, 
each of which point to more configuration files in their respective directories. 
If `train.py` is run without any further options, it will load the default values 
for these three config files.

Outputs from training runs are stored in the `outputs/` directory. If this 
directory is not present, it will be created on first run. Outputs are, by
default, grouped by `experiment.name`, model type, date, and time.
An example `outputs/` directory from the model sepcified in the default
configuration would look something like this:
```
outputs/
  experiment-1/
    SRN-SRN-None/
      2021-01-01_14-30-00/
        .hydra/
        tensorboard/
        model.pt
        train.log
```
The `model.pt` file contains the model weights.
The `train.log` file records the output of `stdout` and `stderr` as logged
during training. The `.hydra/` directory contains a copy of the configuration
specified when training began. The `tensorboard` directory contains the
tensorboard events created during training.

### Overriding defaults
The default values for each configuration option can be overwritten, 
either through the creation of new YAML configuration files and/or
through values provided through the command-line interface. The 
default `train.yaml` looks like the following:
```
defaults:
  - experiment/hyperparameters: default
  - experiment/model: sequence-srn-inattentive
  - experiment/dataset: alice-herself

  - hydra/output: custom

experiment:
  name: experiment-1

pretty_print: True
```
You can (and should) change `experiment.name` to be whatever you want.
The three `experiment/...` parameters under the `defaults:` parameter
are specifying config files in those directories to load. To run a
training instance with a different configuraiton, create new 
config files which match the schema of the provided ones and point
`train.yaml` to look at those files instead. For example, we might
add a new model configuration called `sequence-gru-inattentive.yaml`,
which specifies the following model:
```YAML
# @package _group_
encoder:
  unit: GRU
  type: sequence
  dropout: 0
  num_layers: 1
  hidden_size: 256
  max_length: 0
  embedding_size: 256
  bidirectional: False
decoder:
  unit: GRU
  type: sequence
  dropout: 0
  num_layers: 1
  max_length: 30
  hidden_size: 256
  attention: None
  embedding_size: 256
```
We can then train with this model by changing `- experiment/model:` to be `inattentive-gru-sequence`:
```
defaults:
  - experiment/hyperparameters: default
  - experiment/model: inattentive-gru-sequence
  - experiment/dataset: alice-herself

  - hydra/output: custom

experiment:
  name: experiment-1

pretty_print: True
```

Alternatively, we can train with this new model directly from the command-line by overwriting the default value:
```bash
python train.py experiment/model=inattentive-gru-sequence
```

## Defining an experiment

An experiment consists of several parts, including:
- **Dataset:** The dataset is the training, testing, and evaluation data given 
    to the model and used to evaluate its performance. The dataset for a given
    experiment is defined as an entry in `conf/experiment/dataset/`, and consists
    of a raw input data source, which lives as a TSV file in
    the `data/raw/` directory and a configuration file which specifies how the
    raw data is turned into splits for training, testing, and evaluation, as well
    as any withholding done to create a generalization set or separate tracking
    splits to evaluate performance on a specific subset of the full data.
- **Models:** Models for an experiment are defined by configuration files in the
    `conf/experiment/model/` directory. These configuration files specify the hyperparameters
    of the model, and allow an experiment to quickly and briefly specify the
    types of models to be used.
- **Hyperparameters:** Things like batch size, early stopping, learning rate,
    and so on are defined by config files in the `conf/experiment/hyperparameters/` directory.

### Dataset Configuration

A `dataset` has the following configuration schema:
```YAML
# @package _group_
name: experiment-1 
input: grammar-1.tsv
source_format: sequence
target_format: sequence
overwrite: false
splits:
  train: 80
  test: 10
  val: 10
withholding:
  - ^Alice \w+ herself.*
tracking:
  alice: ^Alice.*
```

This defines the relationship between the input data (the file specified by the 
`input` parameter) and the generated output data. Multiple experiments can use
the same `input` file but may generate distinct outputs due to differences in 
how the splits are generated, different withholding patterns, or different 
tracking patterns. The `splits` parameter defines a dictionary of percentages
which must add up to 100. The percentages specify how the *in-sample* data
is broken into splits. You must provide values for `train`, `test`, and `val`.
The `withholding` parameter specifies a list of regular expressions which
define a generalization set. Input sequences which match any of these regexes
are withheld from the in-sample data and are instead put into a separate split.
In total, the generated `train.pt`, `test.pt`, `val.pt`, and `gen.pt` files
are mutually disjoint and their union is simply the original `input` file.

The `tracking` parameter also specifies various `.pt` files, but in a different
way. This is a dictionary of the form `{name : regex}`, where `name` is the name
of the generated file and `regex` is a regex which determines which sequences
are included. Importantly, this has no effect on split generation and is merely
used to allow for the logging of model performance on different subsets of the
data during training.

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