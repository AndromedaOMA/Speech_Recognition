import torch.nn as nn
import torch.nn.functional as f

from configs.train_configs import TrainConfig


# ================= Model =================
def layernorm(x, input_features, config: TrainConfig):
    Layer = nn.LayerNorm(input_features)
    Layer.to(config.device)
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
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(config.res_cnn_output_channels * config.input_features, config.bi_rnn_input_size)
        self.res_cnn = nn.Sequential(
            *[ResidualCNN(in_channels=config.res_cnn_input_channels,
                          out_channels=config.res_cnn_output_channels,
                          kernel=config.res_cnn_kernel_size,
                          stride=config.res_cnn_stride,
                          padding=config.res_cnn_padding,
                          dropout_probability=config.res_cnn_dropout_prob,
                          input_features=config.input_features)
              for _ in range(config.num_res_cnn)])
        self.rnn_layers = nn.Sequential(
            *[BidirectionalGRU(input_size=config.bi_rnn_input_size if i == 0 else 2 * config.bi_rnn_input_size,
                               hidden_size=config.bi_rnn_hidden_size,
                               num_layers=config.bi_rnn_num_layers,
                               dropout_probability=config.bi_rnn_dropout_prob) for i in range(config.num_rnn)])
        self.classifier = nn.Sequential(
            nn.Linear(config.cl_input_size, config.cl_input_size // 2),
            nn.GELU(),
            nn.Dropout(config.cl_dropout_prob),
            nn.Linear(config.cl_input_size // 2, config.cl_num_classes))

    def forward(self, batch, config: TrainConfig):
        time_dim = batch.shape[2]
        batch_size = batch.shape[0]
        out = self.cnn(batch)
        out = self.res_cnn(out)
        out = out.view(batch_size, time_dim, config.res_cnn_output_channels * config.input_features)
        out = self.linear(out)
        out = self.rnn_layers(out)
        out = self.classifier(out)
        return out

