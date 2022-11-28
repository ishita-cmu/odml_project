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

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}


def show_batch_predictions(batch_data, model):
    """
    Visualizes image, image with actual keypoints and
    image with predicted keypoints.
    Finger colors are in COLORMAP.

    Inputs:
    - batch data is batch from dataloader
    - model is trained model
    """
    inputs = batch_data["image"].to('cuda')
    true_keypoints = batch_data["keypoints"].numpy()
    batch_size = true_keypoints.shape[0]
    pred_keypoints = model(inputs)
    pred_keypoints = pred_keypoints.cpu().detach().numpy()
    # pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
    images = batch_data["image_raw"].numpy()
    images = np.moveaxis(images, 1, -1)

    plt.figure(figsize=[12, 4 * batch_size])
    for i in range(1):
        image = images[i]
        true_keypoints_img = true_keypoints[i] * RAW_IMG_SIZE
        pred_keypoints_img = pred_keypoints[i] * RAW_IMG_SIZE

        plt.subplot(batch_size, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 2)
        plt.imshow(image)
        for finger, params in COLORMAP.items():
            plt.plot(
                true_keypoints_img[np.array(params["ids"])*2],
                true_keypoints_img[(np.array(params["ids"])*2+1)],
                params["color"],
            )
        plt.title("True Keypoints")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 3)
        plt.imshow(image)
        for finger, params in COLORMAP.items():
            plt.plot(
                pred_keypoints_img[np.array(params["ids"])*2],
                pred_keypoints_img[(np.array(params["ids"])*2+1)],
                params["color"],
            )
        plt.title("Pred Keypoints")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig('haath.jpg')

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
            # image_raw_transform = transforms.ToTensor()
            # image_transform = transforms.Compose(
            #     [
            #         transforms.Resize(MODEL_IMG_SIZE),
            #         transforms.CenterCrop(MODEL_IMG_SIZE),
            #         transforms.ToTensor(),
            #         transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            #     ]
            # )
            # # image_name = self.image_names[idx]
            # image_raw = Image.open("/content/test_img.jpg")
            # image = image_transform(image_raw)
            # image = image.reshape(1,3,224,224)
            # output = self.model(image).squeeze().reshape(21,2).detach().numpy()
            # img_test = Image.open("/content/test_img.jpg")
            # image_transform_test = transforms.Compose(
            #     [
            #         transforms.Resize(MODEL_IMG_SIZE),
            #         transforms.CenterCrop(MODEL_IMG_SIZE),
            #         transforms.ToTensor(),
            #         # transforms.Normalize(mean=DATASET_MEANS, std=DATASET_STDS),
            #         transforms.ToPILImage(),
            #     ]
            # )
            # img_test = image_transform_test(img_test)

            # plt.imshow(img_test)
            # plt.scatter(output[:, 0]*224, output[:, 1]*224, c="green", alpha=0.5)
            # plt.savefig(f"test_{epoch}.jpg")

            

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

            #########################
            if i < 3 :
              show_batch_predictions(data, self.model)
            #########################
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
