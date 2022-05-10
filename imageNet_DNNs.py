import os
import tensorflow as tf
from zipfile import ZipFile
import urllib.request
import numpy as np
import scipy.misc
import imageio
#from ..utils import download_dataset
from tensorflow.keras.optimizers import Adam


# from PIL import Image
vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', classes=1000, classifier_activation="softmax")
vgg16_model.summary()
vgg16_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])



_urls = {"http://cs231n.stanford.edu/tiny-imagenet-200.zip": "tiny-imagenet-200.zip"}
_name = "tinyimagenet"

def load(path=None):
    """
    Tiny Imagenet has 200 classes. Each class has 500 training images, 50
    validation images, and 50 test images. We have released the training and
    validation sets with images and annotations. We provide both class labels an
    bounding boxes as annotations; however, you are asked only to predict the
    class label of each image without localizing the objects. The test set is
    released without labels. You can download the whole tiny ImageNet dataset
    here.
    """

    if path is None:
        path = os.environ["DATASET_path"]

    #download_dataset(path, _name, _urls)

    # Loading the file
    f = ZipFile(os.path.join(path, _name, "tiny-imagenet-200.zip"), "r")
    names = [name for name in f.namelist() if name.endswith("JPEG")]
    val_classes = np.loadtxt(
        f.open("tiny-imagenet-200/val/val_annotations.txt"),
        dtype=str,
        delimiter="\t",
    )
    val_classes = dict([(a, b) for a, b in zip(val_classes[:, 0], val_classes[:, 1])])
    x_train, x_test, x_valid, y_train, y_test, y_valid = [], [], [], [], [], []
    for name in names:
        if "train" in name:
            classe = name.split("/")[-1].split("_")[0]
            x_train.append(
                imageio.imread(f.open(name), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )
            y_train.append(classe)
        if "val" in name:
            x_valid.append(
                imageio.imread(f.open(name), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )
            arg = name.split("/")[-1]
            print(val_classes[arg])
            y_valid.append(val_classes[arg])
        if "test" in name:
            x_test.append(
                imageio.imread(f.open(name), pilmode="RGB").transpose(
                    (2, 0, 1)
                )
            )

    dataset = {
        "train_set/images": x_train,
        "train_set/labels": y_train,
        "test_set/images": x_test,
        "valid_set/images": x_valid,
        "valid_set/labels": y_valid,
    }
    return dataset

dataset  = load("/home/sielai/Downloads")
x_test = np.asarray(dataset["train_set/images"])
y_test = np.asarray(dataset["train_set/labels"])

print(x_test.shape)

score = vgg16_model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])