import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


class UNetTrainer:
    def __init__(self, model, device, criterion, optimizer, scheduler):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.best_loss = float('inf')

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def val_epoch(self, dataloader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def train(self, train_loader, val_loader, num_epochs=25):
        since = time.time()

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.val_epoch(val_loader)

            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')

            # 深拷贝模型
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

            self.scheduler.step(val_loss)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {self.best_loss:.4f}')

        # 加载最佳模型权重
        self.model.load_state_dict(self.best_model_wts)
        return self.model