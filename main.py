import torch
from configs.train_configs import TrainConfig
import json
from models.speech_recognition_model import SpeechRecognitionModel

config = TrainConfig()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    configs = TrainConfig()
    with open("config.json", mode="w", encoding="utf-8") as file:
        json.dump(configs.__dict__, file, indent=4)

    model = SpeechRecognitionModel(config).to(device)
