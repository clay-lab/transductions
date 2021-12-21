import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class TPDNDataset(Dataset):
    def collate_fn(data):
        """
        Padds data to make it batchable. Returns tuple of
        (fillers, roles, encodings), where each tensor is of size BxL, where B is
        batch size and L is the maximum length of any entry in the original tuple of tensors
        """

        data = np.array(data, dtype=object)

        f_arr = tuple(data[:, 0])
        r_arr = tuple(data[:, 1])
        a_arr = tuple(data[:, 2])
        e_arr = tuple(data[:, 3])
        t_arr = tuple(data[:, 4])

        f_stack = pad_sequence(f_arr, batch_first=True)
        r_stack = pad_sequence(r_arr, batch_first=True)
        a_stack = pad_sequence(a_arr, batch_first=True)
        e_stack = pad_sequence(e_arr, batch_first=True)
        t_stack = pad_sequence(t_arr, batch_first=True)

        return (f_stack, r_stack, a_stack, e_stack, t_stack)

    def __init__(self, datafile: str):
        super(TPDNDataset, self).__init__()
        self.data = pd.read_pickle(datafile)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fillers = torch.tensor(self.data.at[idx, "fillers"])
        roles = torch.tensor(self.data.at[idx, "roles"])
        annotations = torch.tensor(self.data.at[idx, "annotation"])
        encoding = torch.tensor(self.data.at[idx, "encoding"])
        target = torch.tensor(self.data.at[idx, "target"])

        return (fillers, roles, annotations, encoding, target)
