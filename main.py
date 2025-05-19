import torch
import json

from torch import optim, nn
from tqdm import tqdm

from configs.train_configs import TrainConfig
from model.speech_recognition_model import SpeechRecognitionModel
from data.prepared_data import Train_Loader, Val_Loader
from train.train import Train

import matplotlib.pyplot as plt


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = SpeechRecognitionModel(configs, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=configs.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=configs.lr, steps_per_epoch=int(len(Train_Loader)),
                                              epochs=configs.epochs, anneal_strategy='linear')
    criterion = nn.CTCLoss(blank=28, ).to(device)

    # =============== Train ===============
    train = Train(optimizer, scheduler, model, criterion, device)
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(configs.epochs), total=configs.epochs):
        train_loss = train.fit(Train_Loader)
        val_loss = train.predict(Val_Loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.plot(range(len(val_losses)), val_losses)
    plt.plot(range(len(train_losses)), train_losses)
    plt.legend(['vl', 'tr'])
