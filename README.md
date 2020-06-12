# transductions
Pytorch/Torchtext implementation of seq2seq transductions

The following argument is required:

file name from which data will be accessed (-t, --task): Example: negation

The following arguments are optional:

encoder (-e, --encoder): Default: GRU. Options: GRU, LSTM, SRN, Tree
decoder (-e, --decoder): Default: GRU. Options: GRU, LSTM, SRN, Tree
attention (-a, --attention): Default: None. Options: location, additive, multiplicative, dotproduct
learning rate (-lr, --learning-rate): Default: 0.01. 
hidden size (-hs, --hidden-size): Default: 256.
number of layers for encoder and decoder (-l, --layers): Default: 1.
length limit of decoded sequences (--max-length): Default: 30.
random seed (-rs, --random-seed): Default: None.
patience (-p, --patience): Default: 3
vocabulary to contain the transformation annotation (-v, --vocab): Default: TRG. Options: SRC, TRG.
dropout amount (-do, --dropout): Default: 0.0.
data representation (--input-format): Default 'sequences'. Options: sequences, trees.
number of epochs (-ep, --epochs): Default: 20.
batch size (-b, --batch-size): Default: 5.


Required Command: 
```bash
python main.py --task negation
```

Optional Command: 
```bash
python main.py --task negation --encoder LSTM --decoder LSTM -a location -lr 0.001 --hidden-size 256 -l 3  --max-length 35 -rs 0.43 -p 4 --vocab SRC -do 0.01 --input-format trees -ep 25 -b 4
```

If an optional command is not provided, the default value will be used.


