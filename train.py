import torch
import torch.nn as nn
from model import Unet
from dataset import SegmentationDataSet
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import numpy as np
import cv2
import os

def binary_transform(element):
    if element > 0.5:
        return 255
    else:
        return 0


EPOCHS = 2
LEARNING_RATE = 0.0005
BATCH_SIZE = 16
NUMBER_WORKERS = 2
CLASSES = 1
IMG_DIR = "data/train"
LABEL_DIR = "data/train_masks"
VAL_IMG_DIR = "data/test"
VAL_LABEL_DIR = "data/test_masks"
LOG_FOLDER = "unet_logs"
SAVED_RESULT_FOLDER = "saved_results"



if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Running on the GPU\n")
    else:
        print("Running on the CPU\n")


    #Loading data
    train_data = SegmentationDataSet(IMG_DIR, LABEL_DIR)
    val_data = SegmentationDataSet(VAL_IMG_DIR, VAL_LABEL_DIR)

    train_dataloader = DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUMBER_WORKERS, drop_last = False)
    val_dataloader = DataLoader(dataset = val_data, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUMBER_WORKERS, drop_last = False)

    print("Data has been loaded...\n")

    #Initializing model

    unet_model = Unet(3, CLASSES)
    optimizer = torch.optim.Adam(unet_model.parameters(), lr = LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    print("Model has been initialized...\n")

    writer = SummaryWriter(LOG_FOLDER)

    print("Writing logs to folder: {}\n".format(LOG_FOLDER))

    count = 0
    #train model with train dataset
    for epoch in range(EPOCHS):
        print("Starting training at epoch {}...".format(epoch + 1))

        i = 0
        for data in train_dataloader:
            unet_model.train()
            img, label = data

            img_out = unet_model(img)

            loss = loss_fn(img_out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("Train_loss", loss, count)


            count = count + BATCH_SIZE

            print("Train loss after training", count, "images:", loss.item())

            #save the model and one result image after every 240 pictures are trained
            if (count % (BATCH_SIZE * 15)) == 0:
                res_img = img[0].detach().numpy()
                res_img_out = img_out[0].detach().numpy()
                res_label = label[0].detach().numpy()

                res_img = np.transpose(res_img, (1, 2, 0))

                res_img_out = np.transpose(res_img_out, (1, 2, 0))

                b_trans = np.vectorize(binary_transform)
                res_img_out = b_trans(res_img_out)

                res_label = np.transpose(res_label, (1, 2, 0))

                b_trans = np.vectorize(binary_transform)
                res_label = b_trans(res_label)

                dest_folder = SAVED_RESULT_FOLDER + '/' + str(count)
                os.mkdir(dest_folder)

                cv2.imwrite(dest_folder + '/' + str(count) + "_result_img.jpg", res_img)
                cv2.imwrite(dest_folder + '/' + str(count) + "_result_out.jpg", res_img_out)
                cv2.imwrite(dest_folder + '/' + str(count) + "_result_label.jpg", res_label)


                torch.save(unet_model.state_dict(), "save_models/unet_" + str(count) + ".pt")

            #calculate the accuracy on validation set after every 400 pictures are trained
            if (count % (BATCH_SIZE * 5)) == 0:
                num_corr = 0
                num_pix = 0
                unet_model.eval()
                with(torch.no_grad()):
                    for data in val_dataloader:
                        img, label = data
                        out_put_img = unet_model(img)
                        out_put_img = torch.sigmoid(out_put_img)
                        out_put_img = (out_put_img > 0.7).float()
                        num_corr += (out_put_img == label).sum()
                        num_pix += torch.numel(out_put_img)

                acc = num_corr / num_pix * 100

                writer.add_scalar("Val_accuracy", acc, count)
                print("=============================The accuracy now is", acc.item())

    writer.close()
    # tensorboard --logdir=unet_logs --port=6007

























