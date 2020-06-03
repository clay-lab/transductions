# transductions
Pytorch/Torchtext implementation of seq2seq transductions

This implementation requires a new designation for whether the transformation will be an input to the encoder or decoder. If `--vocab` is set to `SRC` it will be the last element inputted to the encoder. If `--vocab` is set to `TRG` it will be the first element to the decoder. The default is set to `SRC`

The original command: 
```bash
python seq2seq.py --encoder GRU --decoder GRU --task question --attention content --lr 0.001 --hs 256
```
New Command: 
```bash
python seq2seqNEW.py --encoder GRU --decoder GRU --task negation --attention content --lr 0.001 --hs 256 --vocab SRC
```
