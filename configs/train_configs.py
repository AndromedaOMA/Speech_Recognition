import torch
from pydantic import BaseModel


class TrainConfig(BaseModel):
    input_features: int = 128
    train_batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 5
    lr: float = 1e-5

    # ResidualCNN block params
    res_cnn_input_channels: int = 32
    res_cnn_output_channels: int = 32
    res_cnn_kernel_size: int = 3
    res_cnn_stride: int = 1
    res_cnn_padding: int = 1
    res_cnn_dropout_prob: float = 0.1
    num_res_cnn: int = 3

    # BidirectuinalGRU block params
    bi_rnn_input_size: int = 512
    bi_rnn_hidden_size: int = 512
    bi_rnn_num_layers: int = 1
    bi_rnn_dropout_prob: float = 0.1
    num_rnn: int = 5

    # Classifier block params
    cl_input_size: int = bi_rnn_input_size * 2
    cl_dropout_prob: float = 0.1
    cl_num_classes: int = 29  # No. of neurons each represents every possible character of the label for every timestamp dimension [tensor_mex_length:2797]
