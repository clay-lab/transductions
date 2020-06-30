# transductions
A Pytorch/Torchtext implementation of seq2seq transductions


## Running `transductions`

The basic interface for `transductions` is the following:
```bash
$ python main.py {train, test}
```

Passing the `train` argument will allow you to train a new model, while 
issuing the `test` argument will allow you to load a model and test it.

### Training

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

Models will be saved in the `models/` directory under a unique named which is
generated at run-time. The unique name takes the form `TASK-DD-MM-YY-#`, where
`#` is an integer which is incremented on every run. This ensures that all runs
generate unique models with predictable names.

#### Training Options:

* `-e, --encoder`: type of encoder used; one of `GRU`, `SRN`, `LSTM`, or `Tree`. Defaults to `GRU`.
* `-d, --decoder`: type of decoder used; one of `GRU`, `SRN`, `LSTM`, or `Tree`. Defaults to `GRU`.
* `-t, --task`: task model is trained to perform. This corresponds to the name of the data files mentioned earlier. These files must be placed in the `data/` directory.
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

#### A note on `--tokens`:

Input type is string separated by '-' and '_'. The dashes separate individual 
words and accuracy measurements, respectively. This argument will include 
accuracy measures for any token in the vocabulary for a given dataset. If no 
argument is included, only the loss, the token-level accuracy, and the 
length-level accuracy will be calculated.

### Testing

Once you've trained a model, you can load it for testing. There are two main
ways to test a model: file mode and interactive mode.

In file mode, you generate a set of testing files, each with the extention `.test`, and place them in the data directory. You can then pass the names of these files to the model, and the model will be run on each of the files and generate a new output file for each task with its predictions. These test
files should have the same structure as the training files.
So, for example, if you had the following setup:

```
.
├──models/
│  └── negation-29-06-20-0/
└──data/
   ├── neg-1.test
   └── neg-2.test

```

You can test model `negation-29-06-20-0` on the `neg-1` and `neg-2` datasets by issuing the following command:

```bash
$ python main.py test -m negation-29-06-20-0 -t neg-1 neg-2
```

Here, `-m, --model` loads the model and `-t, --task` specifies the task(s)
to evaluate the model on. This will generate two new files,
`results/negation-29-06-20-0-neg-1` and `results/negation-29-06-20-0-neg-2`,
which contain the model's output on these datasets.

In interactive mode, you don't specify any tasks. Instead you enter a 
Read-Evaluate-Print Loop (REPL) which loads your model and prompts you to enter
sequences into a prompt and the model's output will be printed below. Sequences
are entered in the following format:
```
> TRANFORMATION Your sequence goes here
```
where `TRANSFORMATION` is the same token as in the training files. So, to 
interactively test the model we just defined, we could issue the command
``` bash
$ python main.py test -m negation-29-06-20-0
```

and we would get the following output:
```
Enter sequences into the negation-29-06-20-0 model for evaluation:
>
```
