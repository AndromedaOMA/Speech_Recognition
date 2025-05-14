import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as f


class Train:
    def __init__(self, optimizer, scheduler, model, criterion, device):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model = model
        self.criterion = criterion
        self.device = device

    def fit(self, train_loader):
        losses = []
        for idx, batch in tqdm(enumerate(train_loader), desc='traning', total=len(train_loader)):
            train_x = batch['data'].to(self.device)
            train_y = batch['label'].to(self.device)
            train_x_len = batch['data_len'].to(self.device)
            train_y_len = batch['label_len'].to(self.device)
            out = self.model(train_x)
            out = f.log_softmax(out, dim=-1)
            out = out.transpose(0, 1)
            loss = self.criterion(out, train_y, train_x_len, train_y_len)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.zero_grad()
            self.optimizer.step()
            self.scheduler.step()
        return np.mean(losses)

    def predict(self, val_loader):
        losses = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(val_loader), desc='validating', total=len(val_loader)):
                val_x = batch['data'].to(self.device)
                val_y = batch['label'].to(self.device)
                val_x_len = batch['data_len'].to(self.device)
                val_y_len = batch['label_len'].to(self.device)
                out = self.model(val_x)
                out = f.log_softmax(out, dim=-1)
                out = out.transpose(0, 1)
                loss = self.criterion(out, val_y, val_x_len, val_y_len)
                losses.append(loss.item())
            return np.mean(losses)
