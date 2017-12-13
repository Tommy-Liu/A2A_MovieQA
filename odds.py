# To be implemented
import numpy as np
from torch.utils.data import Dataset

from config import MovieQAConfig

config = MovieQAConfig()


# To be implemented
class EmbeddingDataSet(Dataset):
    def __init__(self, num_given=0, use_length=12):
        self.load = {
            'vec': np.load(config.encode_embedding_vec_file),
            'word': np.load(config.encode_embedding_key_file),
            'len': np.load(config.encode_embedding_len_file)
        }

    def __len__(self):
        return len(self.load['len'])

    def __getitem__(self, idx):
        return self.load['vec'][idx], self.load['word'][idx], self.load['len'][idx]
