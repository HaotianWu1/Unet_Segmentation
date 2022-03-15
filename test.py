import os
from random import sample
import shutil


if __name__ == "__main__":

    #
    # img_names = os.listdir('data/train')
    #
    # for i in range(len(img_names)):
    #     img_names[i] = img_names[i].split(".")[0]
    #
    #
    # test_names = sample(img_names,500)
    #
    # for i in test_names:
    #     img = i + '.jpg'
    #     label = i + '_mask.jpg'
    #     shutil.move('data/train/' + img, 'data/test/' + img)
    #     shutil.move('data/train_masks/' + label, 'data/test_masks/' + label)

