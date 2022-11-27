import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

N_KEYPOINTS = 21
N_IMG_CHANNELS = 3
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 224
DATASET_MEANS = [0.3950, 0.4323, 0.2954]
DATASET_STDS = [0.1966, 0.1734, 0.1836]
MODEL_NEURONS = 16


class Trainer:
    def __init__(self, model, criterion, optimizer, config, scheduler=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss = {"train": [], "val": []}
        self.epochs = config["epochs"]
        self.batches_per_epoch = config["batches_per_epoch"]
        self.batches_per_epoch_val = config["batches_per_epoch_val"]
        self.device = config["device"]
        self.scheduler = scheduler
        self.checkpoint_frequency = 100
        self.early_stopping_epochs = 10
        self.early_stopping_avg = 10
        self.early_stopping_precision = 5

    def train(self, train_dataloader, val_dataloader):
        for epoch in range(self.epochs):
            # print("yo")
            ### generate_test_img
            image_raw_transform = transforms.ToTensor()
            image_transform = transforms.Compose(
                [
                    transforms.Resize(MODEL_IMG_SIZE),
                    transforms.CenterCrop(MODEL_IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
                ]
            )
            # image_name = self.image_names[idx]
            image_raw = Image.open("/content/test_img.jpg")
            image = image_transform(image_raw)
            image = image.reshape(1,3,224,224)
            output = self.model(image).squeeze().reshape(21,2).detach().numpy()
            img_test = Image.open("/content/test_img.jpg")
            image_transform_test = transforms.Compose(
                [
                    transforms.Resize(MODEL_IMG_SIZE),
                    transforms.CenterCrop(MODEL_IMG_SIZE),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
                    transforms.ToPILImage(),
                ]
            )
            img_test = image_transform_test(img_test)

            plt.imshow(img_test)
            plt.scatter(output[:, 0]*224, output[:, 1]*224, c="green", alpha=0.5)
            plt.savefig(f"test_{epoch}.jpg")

            self._epoch_train(train_dataloader)
            self._epoch_eval(val_dataloader)
            print(
                "Epoch: {}/{}, Train Loss={}, Val Loss={}".format(
                    epoch + 1,
                    self.epochs,
                    np.round(self.loss["train"][-1], 10),
                    np.round(self.loss["val"][-1], 10),
                )
            )

            # reducing LR if no improvement
            if self.scheduler is not None:
                self.scheduler.step(self.loss["train"][-1])

            # saving model
            if (epoch + 1) % self.checkpoint_frequency == 0:
                torch.save(
                    self.model.state_dict(), "model_{}".format(str(epoch + 1).zfill(3))
                )

            # early stopping
            if epoch < self.early_stopping_avg:
                min_val_loss = np.round(np.mean(self.loss["val"]), self.early_stopping_precision)
                no_decrease_epochs = 0

            else:
                val_loss = np.round(
                    np.mean(self.loss["val"][-self.early_stopping_avg:]), 
                                    self.early_stopping_precision
                )
                if val_loss >= min_val_loss:
                    no_decrease_epochs += 1
                else:
                    min_val_loss = val_loss
                    no_decrease_epochs = 0
                    #print('New min: ', min_val_loss)

            if no_decrease_epochs > self.early_stopping_epochs:
                print("Early Stopping")
                break

        torch.save(self.model.state_dict(), "model_final")
        return self.model

    def _epoch_train(self, dataloader):
        self.model.train()
        running_loss = []

        for i, data in enumerate(dataloader, 0):
            inputs = data["image"].to(self.device)
            labels = data["keypoints"].to(self.device)
            # print("enter here")
            # print(inputs.shape)
            # print(labels.shape)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            
            # print(outputs.shape)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss.append(loss.item())

            if i == self.batches_per_epoch:
                epoch_loss = np.mean(running_loss)
                self.loss["train"].append(epoch_loss)
                break

    def _epoch_eval(self, dataloader):
        self.model.eval()
        running_loss = []

        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs = data["image"].to(self.device)
                labels = data["keypoints"].to(self.device)

                outputs = self.model(inputs)
                # print(outputs.shape)
                # print(labels.shape)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.batches_per_epoch_val:
                    epoch_loss = np.mean(running_loss)
                    self.loss["val"].append(epoch_loss)
                    break
