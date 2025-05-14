import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

import itertools


class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_features = 128
    train_batch_size = 8
    val_batch_size = 8
    epochs = 5
    lr = 1e-5

    # ResidualCNN block params
    res_cnn_input_channels = 32
    res_cnn_output_channels = 32
    res_cnn_kernel_size = 3
    res_cnn_stride = 1
    res_cnn_padding = 1
    res_cnn_dropout_prob = 0.1
    num_res_cnn = 3

    # BidirectuinalGRU block params
    bi_rnn_input_size = 512
    bi_rnn_hidden_size = 512
    bi_rnn_num_layers = 1
    bi_rnn_dropout_prob = 0.1
    num_rnn = 5

    # Classifier block params
    cl_input_size = bi_rnn_input_size * 2
    cl_dropout_prob = 0.1
    cl_num_classes = 29  # No. of neurons each represents every possible character of the label for every timestamp dimension [tensor_mex_length:2797]


# ================== Data ==================
train_data = torchaudio.datasets.LIBRISPEECH(root='../', url='dev-clean', download=True)
val_data = torchaudio.datasets.LIBRISPEECH(root='../', url='test-clean', download=True)
print(f"next(t_l)['data'].shape: waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id:: {train_data[1]}")

train_transforms = nn.Sequential(  # (batch ,channel, feature, time)
    torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=Config.input_features),  # Coverts raw audio waveform into a Mel-scaled spectogram
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

print(f"tensor_mex_length:{tensor_mex_length}, label_max_length: {label_max_length}")

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


Train_Loader = DataLoader(LibriSpeechDataset(data=train_data, transform=train_transforms),
                          batch_size=Config.train_batch_size,
                          shuffle=True)
Val_Loader = DataLoader(LibriSpeechDataset(data=val_data, transform=val_transforms),
                        batch_size=Config.val_batch_size,
                        shuffle=True)

t_l = iter(Train_Loader)
print(next(t_l)['data'].shape)


# ================= Model =================
def layernorm(x, input_features):
    Layer = nn.LayerNorm(input_features)
    Layer.to(Config.device)
    return Layer(x)


# ======================= ResidualCNN =======================
class ResidualCNN(nn.Module):
    def __init__(self, input_features, in_channels, out_channels, kernel, stride, padding, dropout_probability):
        super().__init__()
        self.input_features = input_features
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)
        self.drop_1 = nn.Dropout(dropout_probability)
        self.drop_2 = nn.Dropout(dropout_probability)

    def forward(self, batch):
        residue = batch
        batch = f.gelu(layernorm(batch, self.input_features))
        batch = self.drop_1(batch)
        batch = self.conv_1(batch)
        batch = f.gelu(layernorm(batch, self.input_features))
        batch = self.drop_2(batch)
        batch = self.conv_2(batch)
        return batch + residue


# ======================= BidirectionalGRU =======================
class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_probability):
        super().__init__()
        self.input_size = input_size
        self.bi_gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, batch):
        batch = f.gelu(layernorm(batch, self.input_size))
        batch = self.bi_gru(batch)
        batch = self.drop(batch[0] if type(batch) == tuple else batch)
        return batch


# =======================SpeechRecognitionModel =======================
class SpeechRecognitionModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(Config.res_cnn_output_channels * Config.input_features, Config.bi_rnn_input_size)
        self.res_cnn = nn.Sequential(
            *[ResidualCNN(in_channels=Config.res_cnn_input_channels,
                          out_channels=Config.res_cnn_output_channels,
                          kernel=Config.res_cnn_kernel_size,
                          stride=Config.res_cnn_stride,
                          padding=Config.res_cnn_padding,
                          dropout_probability=Config.res_cnn_dropout_prob,
                          input_features=Config.input_features)
              for _ in range(Config.num_res_cnn)])
        self.rnn_layers = nn.Sequential(
            *[BidirectionalGRU(input_size=Config.bi_rnn_input_size if i == 0 else 2 * Config.bi_rnn_input_size,
                               hidden_size=Config.bi_rnn_hidden_size,
                               num_layers=Config.bi_rnn_num_layers,
                               dropout_probability=Config.bi_rnn_dropout_prob) for i in range(Config.num_rnn)])
        self.classifier = nn.Sequential(
            nn.Linear(Config.cl_input_size, Config.cl_input_size // 2),
            nn.GELU(),
            nn.Dropout(Config.cl_dropout_prob),
            nn.Linear(Config.cl_input_size // 2, Config.cl_num_classes))

    def forward(self, batch):
        time_dim = batch.shape[2]
        batch_size = batch.shape[0]
        out = self.cnn(batch)
        out = self.res_cnn(out)
        out = out.view(batch_size, time_dim, Config.res_cnn_output_channels * Config.input_features)
        out = self.linear(out)
        out = self.rnn_layers(out)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    model = SpeechRecognitionModel().to(Config.device)
