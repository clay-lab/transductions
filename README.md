# transductions

A Pytorch/Torchtext framework for building and testing Sequence-to-Sequence
models and experiments. Transductions relies heavily on 
[Hydra](https://github.com/facebookresearch/hydra) for configuration,
allowing you to specify model architectures, datasets, and experiments using
YAML files for easy modularity and reproducibility.

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


## Submitting files on GRACE

`transductions` comes with a bash script to automate the deployment of a testing run to the GRACE cluster at Yale. Once logged into your GRACE account,
issue the following command.

```
./batcher.sh EXPDIR TASK ENC DEC ATTN
```

This will generate the SBATCH script `TASK-ENC-DEC-ATTN.sh` and then run `sbatch TASK-ENC-DEC-ATTN.sh`, submitting the job. You can (and should) configure this script before using it to suit your needs. In particular, please change the email address.

