import torch
from torch.utils.data import Dataset, DataLoader
from utils import load_tokens, Config, get_files
import random

class TrainingDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_length = None
        raw_tokens = load_tokens(self.cfg, verbose=False)
        self.process_raw_tokens(raw_tokens)

    def process_raw_tokens(self, rtokens):
        # given raw tokens, create x, y
        self.data = []
        for t in rtokens:
            x = t[0:-1]
            y = t[1:]
            self.data.append([x, y])

    def __len__(self):
        if self.data_length is None:
            count = 0
            for d in self.data:
                count += len(d[0]) - self.cfg.seq_length
            self.data_length = count

        return self.data_length


    def __getitem__(self, idx):
        count = 0
        for d in self.data:
            d_count = len(d[0]) - self.cfg.seq_length

            if count + d_count > idx:
                idx_d = idx - count
                x = d[0][idx_d:idx_d+self.cfg.seq_length]
                y = d[1][idx_d:idx_d + self.cfg.seq_length]
                return [x, y]

            count += d_count

        raise TypeError(f'Failed to get training item with index {idx:,}')
        return None

class TrainingDataloader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        self.pos = [random.randint(0, len(self.dataset)-1) for _ in range(self.batch_size)]
        return self

    def __next__(self):
        self.pos = [random.randint(0, len(self.dataset)-1) for _ in range(self.batch_size)]
        next_data = [self.dataset[p] for p in self.pos]
        next_data = torch.as_tensor(next_data, dtype=torch.int64)
        return next_data

def save_training_state(cfg, model, optimizer, batch):
    fpath = f'{cfg.SAVED_MODELS_FOLDER}{cfg.model_name}_{int(batch)}.mdl'

    torch.save({
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, fpath)

def get_latest_file_id(mfiles):
    max_id =0
    for m in mfiles:
        m = int(m.split('\\')[-1].split('_')[-1].split('.')[0])
        if m>max_id:
            max_id = m

    return str(max_id)
def load_training_state(cfg, model, optimizer):
    mfiles = get_files(directory=f'{cfg.SAVED_MODELS_FOLDER}', filter=f'{cfg.model_name}_*.mdl')

    if len(mfiles) == 0:
        return 0, model, optimizer

    latest_id = get_latest_file_id(mfiles)
    fpath = f'{cfg.SAVED_MODELS_FOLDER}{cfg.model_name}_{latest_id}.mdl'

    checkpoint = torch.load(fpath)
    batch = checkpoint['batch']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f'Using saved model {fpath}, batch: {batch}')

    return batch, model, optimizer

def train_loop(cfg, train_dataloader, model, optimizer):

    batch, model, optimizer = load_training_state(cfg, model, optimizer)
    batch += 1

    # set the model to training mode - important for batch normalization and dropout layers
    model.train()
    for data in train_dataloader:
        x = data[:, 0, :]
        y = data[:, 1, :]

        # compute prediction and loss
        # x is (batch_size, seq_length)
        _, loss = model(x, y)

        # backpropagation
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():>7f}")

        save_training_state(cfg, model, optimizer, batch)
        batch += 1


