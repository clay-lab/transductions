import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torchtext
import torch


sns.set()

def plot_from_batch(model,batch,ex_num):
    outputs, final_hidden = model.encoder(batch.src)
    decoder_out, _, attention = model.decoder(batch.trg, final_hidden, outputs, batch.src, 0)
    #cols = [model.output_vocab.itos[x] for x in decoder_out.argmax(dim=2)[:,ex_num]]
    #inds = [model.input_vocab.itos[x] for x in batch.src[:,ex_num]]
    out = [model.output_vocab.itos[x] for x in decoder_out.argmax(dim=2)[:,ex_num]]
    out_len = out.index('<eos>') if '<eos>' in out else len(out)
    out = out[:out_len] 
    source = [model.input_vocab.itos[x] for x in batch.src[:,ex_num]]
    source_len = source.index('<eos>')
    source = source[:source_len] 
    plt.figure(figsize=(15, 15))
    patt = pd.DataFrame(attention[:out_len,ex_num,1:source_len-1].detach().numpy().T, index=source[1:-1], columns=out)
    sns.heatmap(patt,square=True, linewidths=.1, annot=True)

def translate_batch(model,batch):
    outputs, final_hidden = model.encoder(batch.src)
    decoder_out, _, _ = model.decoder(batch.trg, final_hidden, outputs, batch.src, 0.0)
    for i in range(batch.src.shape[1]):
        output = [model.output_vocab.itos[x] for x in decoder_out.argmax(dim=2)[:,i]]
        output = ' '.join(output[:output.index('<eos>')] if '<eos>' in output else output)
        source = [model.input_vocab.itos[x] for x in batch.src[1:,i]]
        source = source[:source.index('<eos>')]
        transform = source[-1]
        source = ' '.join(source[:-1])
        target = [model.output_vocab.itos[x] for x in batch.trg[1:,i]]
        target = ' '.join(target[:target.index('<eos>')])
        print(i, source, '-'+transform+'->',output)
        print("target:",target)

def plot(model, sentence: str, trans: str):
    source = prepare_input(model,sentence,trans)
    target = prepare_target(model,sentence)
    outputs, final_hidden = model.encoder(source)
    decoder_out, _, attention = model.decoder(target, final_hidden, outputs, source, 0)

    out = [model.output_vocab.itos[x] for x in decoder_out.argmax(dim=2)[:,0]]
    out_len = out.index('<eos>') if '<eos>' in out else len(out)
    out = out[:out_len] 
    source = [model.input_vocab.itos[x] for x in source[:,0]]
    source_len = source.index(trans)
    source = source[1:source_len] 
    plt.figure(figsize=(15, 15))
    patt = pd.DataFrame(attention[:out_len,0,1:source_len].detach().numpy().T, index=source, columns=out)
    sns.heatmap(patt,square=True, linewidths=.1,annot=True)

def translate(model, sentence: str, trans):
    source = prepare_input(model,sentence,trans)
    target = prepare_target(model,sentence)
    outputs, final_hidden = model.encoder(source)
    decoder_out, _, _ = model.decoder(target, final_hidden, outputs, source, 0.)
    
    output = [model.output_vocab.itos[x] for x in decoder_out.argmax(dim=2)[:,0]]
    output = ' '.join(output[:output.index('<eos>')] if '<eos>' in output else output)
    source = [model.input_vocab.itos[x] for x in source[1:,0]]
    source = ' '.join(source[:source.index(trans)])
    print(source, '->', output)

def prepare_input(model, source:str, trans:str):
    source = [source]
    bos, eos = model.input_vocab.stoi['<bos>'], model.input_vocab.stoi['<eos>']
    t_index = model.input_vocab.stoi[trans] 
    source = [s.split() for s in source]
    source = list(map(lambda s: list([model.input_vocab.stoi[x] for x in s]), source))
    source = [[bos] + s + [t_index] + [eos] for s in source]
    source = torch.tensor(source, dtype=torch.long).T
    return source

def prepare_target(model, source:str):
    source = [source]
    s_length = len(source[0].split())
    bos_out, eos_out, unk_out = model.output_vocab.stoi['<bos>'], model.output_vocab.stoi['<eos>'], model.output_vocab.stoi['<unk>']    
    target = [[bos_out] + [unk_out] * s_length + [eos_out]] * len(source)
    target  = torch.tensor(target, dtype = torch.long).T
    return target


