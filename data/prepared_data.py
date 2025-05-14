import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from configs.train_configs import TrainConfig
import itertools

config = TrainConfig()

# ================== Data ==================
train_data = torchaudio.datasets.LIBRISPEECH(root='../../', url='dev-clean', download=True)
val_data = torchaudio.datasets.LIBRISPEECH(root='../../', url='test-clean', download=True)
# print(f"next(t_l)['data'].shape: waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id:: {train_data[1]}")

train_transforms = nn.Sequential(  # (batch ,channel, feature, time)
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=config.input_features),  # Coverts raw audio waveform into a Mel-scaled spectogram
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),  # Randomly masks frequency bands in the Mel spectogram
    torchaudio.transforms.TimeMasking(time_mask_param=35)  # Randomly m,asks time steps in the Mel spectogram
)
val_transforms = nn.Sequential(  # (batch ,channel, feature, time)
    torchaudio.transforms.MelSpectrogram()
)

# ========= Searching for the longest data and label =========
all_data = itertools.chain(train_data, val_data)

tensor_mex_length = -1
label_max_length = -1
for instance in all_data:
    data_len = train_transforms(instance[0]).shape[-1]
    label_len = len(instance[2])

    if data_len > tensor_mex_length:
        tensor_mex_length = data_len
    if label_len > label_max_length:
        label_max_length = label_len

# print(f"tensor_mex_length:{tensor_mex_length}, label_max_length: {label_max_length}")

# ======================= Alpha Encoder =======================
alphas = ["'", ' ',
          'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
          'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
nums = range(28)
alpha_to_num_dict = {alphas[i]: nums[i] for i in range(len(alphas))}
num_to_alpha_dict = {v: k for k, v in alpha_to_num_dict.items()}


def num_to_alpha(number):
    """Decoding"""
    return num_to_alpha_dict[number.item()]


def alpha_to_num(alpha):
    """Encoding"""
    return alpha_to_num_dict[alpha.lower()]


# ================== Transform data ==================
class LibriSpeechDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __getitem__(self, idx):
        audio_raw, _, label, _, _, _ = self.data[idx]
        audio_tensor = self.transform(audio_raw)
        padded_audio = f.pad(audio_tensor, (0, tensor_mex_length - audio_tensor.shape[2]), mode='constant', value=0)
        label_tensor = torch.tensor(list(map(alpha_to_num, label)), dtype=torch.long)
        padded_label = f.pad(label_tensor, (0, label_max_length - label_tensor.shape[0]), mode='constant', value=0)
        return {
            'data': padded_audio.transpose(-1, -2),
            'label': padded_label,
            'data_len': torch.tensor(audio_tensor.shape[2], dtype=torch.long),
            'label_len': torch.tensor(label_tensor.shape[0], dtype=torch.long)
        }

    def __len__(self):
        return len(self.data)


# ================== Loaders ==================
Train_Loader = DataLoader(LibriSpeechDataset(data=train_data, transform=train_transforms),
                          batch_size=config.train_batch_size,
                          shuffle=True)
Val_Loader = DataLoader(LibriSpeechDataset(data=val_data, transform=val_transforms),
                        batch_size=config.val_batch_size,
                        shuffle=True)

# t_l = iter(Train_Loader)
# print(next(t_l)['data'].shape)

