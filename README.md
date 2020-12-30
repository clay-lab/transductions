# transductions

A Pytorch/Torchtext framework for building and testing Sequence-to-Sequence
models and experiments. Transductions relies heavily on 
[Hydra](https://github.com/facebookresearch/hydra) for configuration,
allowing you to specify model architectures, datasets, and experiments using
YAML files for easy modularity and reproducibility.

## Training

To train a new model, run the following script:
```bash
python train.py
```
The `train.py` script is configured through the `train.yaml` file in the `config`
directory. This YAML file specifies three further configuration options: 
`hyperparameters`, `model`, and `experiment`, each of which point to more 
configuration files in their respective directories. If `train.py` is run without
any further options, it will load the default values for these three config
files.

Outputs from training runs are stored in the `outputs/` directory. If this 
directory is not present, it will be created on first run. Each run gets its own
folder, sorted in a two-part hierarchy of `DATE` and `TIME`. An example output
directory will look something like this:
```
outputs/
  2021-01-01/
    08-30-00/
      .hydra/
      model.pt
      source.pt
      target.pt
      train.log
```
The `model.pt` file contains the model weights. `source.pt` and `target.pt`
contain the source and target fields needed to instantiate vocabularies.
The `train.log` file records the output of `stdout` and `stderr` as logged
during training.

### Overriding defaults
To customize the training runs, add new YAML files in the `experiment`, 
`hyperparameters`, and `model` directories. You can then load these at runtime
by providing names to the training script:
```
python main.py experiment=EXPERIMENT_NAME model=MODEL_NAME hyperparameteres=HYP_NAME
```
These would load the `experiment/EXPERIMENT_NAME.yaml`, 
`model/MODEL_NAME.yaml`, and `hyperparameteres/HYP_NAME.yaml`, respectively.

## Defining an experiment

An experiment consists of several parts, including:
- **Dataset:** The dataset is the training, testing, and evaluation data given 
    to the model and used to evaluate its performance. The dataset for a given
    experiment consists of a raw input data source, which lives as a TSV file in
    the `data/raw/` directory and a configuration file which specifies how the
    raw data is turned into splits for training, testing, and evaluation, as well
    as any withholding done to create a generalization set or separate tracking
    splits to evaluate performance on a specific subset of the full data.
- **Models:** Models for an experiment are defined by configuration files in the
    `config/model` directory. These configuration files specify the hyperparameters
    of the model, and allow an experiment to quickly and briefly specify the
    types of models to be used.
- **Training Parameters:** Things like batch size, early stopping, learning rate,
    and so on are defined by config files in the `config/training/` directory.

### Dataset Configuration

An experiment has the following configuration schema:
```YAML
experiment:
  name: experiment-1 
  dataset:
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

## Running an experiment

Running an experiment involves specifying three run-time Hydra configuration 
parameters: `experiment`, `model`, and `training`. The `experiment` parameter
defines the dataset, including the location of the raw data and the split 
configuration, along with the metrics which are tracked during and after 
training. The `model` parameter specifies the network hyperparameters, including
things like `unit_type`, `hidden_size`, and so on. The `training` parameter 
specifies the training hyperparameters like learning rate, batch size, and the 
number of epochs. These parameters each correspond to to subdirectories of the
`config/` directory where named YAML configuration files should live. These
files can then be passed in as run-time arguments as follows:
```
$ python main.py experiment=EXPERIMENT_NAME model=MODEL_NAME training=TRAINING_NAME
```
which would load values specified in `config/experiment/EXPERIMENT_NAME.yaml` 
and so on.
