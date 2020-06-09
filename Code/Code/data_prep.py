import pandas as pd
import numpy as np
import torchtext.data as tt

VAL_RATIO = 0.2

def prepare_csv(filename, seed=999):
    df_train = pd.read_csv("data/"+filename+".csv")
    idx = np.arange(df_train.shape[0])
    #np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/"+filename+"_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        "cache/"+filename+"_val.csv", index=False)
    df_test = pd.read_csv("data/"+filename+"_test.csv")
    df_test.to_csv("cache/"+filename+"_test.csv", index=False)




def load_and_prepare_dataset(filename, batch_size, min_freq=1, lower=False):
    lower = True
    prepare_csv(filename)
    SRC = tt.Field(init_token="<bos>", eos_token="<eos>", lower=True)
    TRG = tt.Field(init_token="<bos>", eos_token="<eos>", lower=True)
#    TRANS = tt.Field(init_token=None, eos_token=None, lower=True)
    train_data, valid_data, test_data = tt.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train=filename+'_train.csv', validation=filename+'_val.csv',
        test = filename+'_test.csv',
        fields=(('src', SRC),# ('trans', TRANS), 
                ('trg', TRG)))
    SRC.build_vocab(train_data.src, min_freq=min_freq)
    TRG.build_vocab(train_data.trg, #train_data.trans, 
                    min_freq=min_freq)
#    TRANS.build_vocab(train_data.trg,train_data.trans, min_freq=min_freq)
    print(f"Size of SRC vocabulary: {len(SRC.vocab)}")
    print(f"Size of TRG vocabulary: {len(TRG.vocab)}")
    iters = tt.BucketIterator.splits((train_data, valid_data, test_data),
                                     sort_key=lambda x: len(x.src),
                                     batch_size=batch_size)



    return iters + (SRC, TRG)