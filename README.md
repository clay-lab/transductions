# transductions
A Pytorch/Torchtext implementation of seq2seq transductions


## Running `transductions`

The basic interface for `transductions` is the following:
```bash
$ python main.py {train, test, log}
```

Passing the `train` argument will allow you to train a new model, while 
issuing the `test` argument will allow you to load a model and test it. 
Both training and testing produce log files, which can be desplayed with the
`log` argument.Note that Python 3 is required; Python 2.7 is not supported.

### Directory structure and shared arguments

A single dataset defines an experiment, a collection of training, testing,
and validation data. In a single experiment, one may test different model
structures, and in turn one can create multiple instances of a particular
structure. This hierarchy defines the directory structure which `transductions`
expects.

The root folder for a collection of experiments is the `experiments` directory,
which may be located anywhere on disk. Preferably, this directory should itself
be a separate git repository. The location of this directory is passed with the
`-E, --expdir` flag on the command line. The structure of the `experiments`
directory is as follows:

```
tree experiments -L 5
.
└── task-name
    ├── SRN-SRN-None
    │   ├── model-1
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-2
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-3
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   └── model-4
    │       ├── SRC.vocab
    │       ├── TRG.vocab
    │       ├── checkpoint.pt
    │       ├── logs
    │       │   └── training
    │       ├── model.pt
    │       └── results
    │           └── task-name.tsv
    ├── SRN-SRN-multiplicative
    │   ├── model-1
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   ├── training
    │   │   │   └── task-name
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-2
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-3
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-4
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   ├── model-5
    │   │   ├── SRC.vocab
    │   │   ├── TRG.vocab
    │   │   ├── checkpoint.pt
    │   │   ├── logs
    │   │   │   └── training
    │   │   ├── model.pt
    │   │   └── results
    │   │       └── task-name.tsv
    │   └── model-6
    │       ├── SRC.vocab
    │       ├── TRG.vocab
    │       ├── checkpoint.pt
    │       ├── logs
    │       │   └── training
    │       ├── model.pt
    │       └── results
    │           └── task-name.tsv
    └── data
        ├── task-name.forms
        ├── task-name.test
        ├── task-name.train
        └── task-name.val
```

`task-name` is the name of the single experiment currently inside this 
experiments directory. Within `task-name` directory are two `structure`
directories, `SRN-SRN-None` and `SRN-SRN-multiplicative`, which in turn
contain `model` directories.

In order to train a model, you will first need a dataset consiting of three
files: `task.train`, `task.test`, and `task.val`. These files should each be
tab-separated values (TSV) of the following form:

```
source	transformation	target
..... 	............. 	....
```
Note that the header (`source	transformation	target`) must be present. The
`source` and `target` columns consist of the input and output data you want to
train the model on, respectively, while the `transformation` column consists of
a transformation token describing what transformation you want the learn to 
perform for that `source`-`target` pair.

No other directories (`structure`, or `model`) need be created as 
`transductions` will automatically create them if necessary.

### Training a model

Models are trained with the following command:

``` bash
python main.py train -E EXP_DIR -t TASK_NAME -e ENCODER -d DECODER 
```

Optionally, you may specify other parts of the model using the following
additional flags:

* `-a, --attention`: type of attention used; one of `none`, `additive`, `multiplicative`, `location`. Defaults to `none`.
* `-lr, --learning-rate`: learning rate of the model. Defaults to `0.01`.
* `-l, --layers`: number of layers in the encoder and decoder. Defaults to `1`.
* `-p, --patience`: number of changes model has to improve loss by `DELTA` to avoid early stopping. Defaults to `3`.
* `-dt, --delta`: amount model needs to improve by to avoid early stopping. Defaults to `0.005`.
* `-v, --vocab`: which vocabulary contains the transformation annotation; one of `SRC` or `TRG`. Defaults to `TRG`.
* `-do, --dropout`: how much dropout to use. Defaults to `0.0`.
* `-ep, --epochs`: number of epochs to train for. Defaults to `40`.
* `-b, --batch-size`: batch size. Defaults to `5`.
* `-sa, --sentacc`: whether or not to log sentence accuracy on validation data. Defaults to `True`.
* `-la, --lengacc`: whether or not to log length accuracy on validation data. Defaults to `True`.
* `-ta, --tokenacc`: whether or not to log token accuracy on validation data. Defaults to `True`.
* `-to, --tokens`: tokens to track the accuracy of on validation data. Defaults to `None`.

During training, loss statistics are calculated for the validation dataset
and stored inside the `logs/training` directory, in the form of a `cox` store.
These logs, and others, can be loaded later.

#### A note on `--tokens`:

Input type is string separated by `-` and `_`. The dashes separate individual 
words and accuracy measurements, respectively. This argument will include 
accuracy measures for any token in the vocabulary for a given dataset. If no 
argument is included, only the loss, the token-level accuracy, and the 
length-level accuracy will be calculated.

### Testing a model

Once you've trained a model, you can load it for testing. There are two main
ways to test a model: file mode and REPL mode.

In REPL mode, you load the model for testing and pass in sequences on the 
command line. The sequences are evaluated and printed back out. This is useful
for one-off testing. To test a model in REPL mode, issue the following command:

```
python main.py test -E EXP_DIR -t TASK_NAME -S ENC-DEC-ATTN -m MODEL-#
```

This will load the following prompt on screen:

```
Enter sequences into the MODEL-# model for evalutaion.
> 
```

Enter sequences in the following format:

```
> TRANSFORMATION This is the sequence here
```

where `TRANSFORMATION` is the transformation token found in the splits file.

To exit the REPL, enter `quit`.


In file mode, you generate a set of testing files to be placed inside the
`data/` directory of the experiment. You then test the model's performance
on any of these testing files. To test a model in file mode, issue the 
following command:

```
python main.py test -E EXP_DIR -t TASK_NAME -S ENC-DEC-ATTN -m MODEL-# -f FILE-1 FILE-2 ...
```

where `FILE-N` corresponds to `data/FILE-N.test`. Accuracy statistics will be
printed out to the screen for each file, and logs for each will be created
inside new `MODEL-#/logs/FILE-N/...` directories.

### Viewing a model's logs

In order to view logs from a previous training or testing run, issue the 
following command:

```
python main.py log -E EXP_DIR -t TASK_NAME -S ENC-DEC-ATTN -m MODEL-# -l LOG
```

This will print out the contents of `MODEL-#/logs/LOG` to the screen. The `-l, 
--log` argument will default to `training` if no log is specified, since this 
log always exists.

