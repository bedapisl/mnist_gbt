import argparse
import Augmentor
from collections import namedtuple
import gzip
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import interpolation
import struct
import time


Dataset = namedtuple("Dataset", ["images", "labels", "height", "width"])


def parse_args():
    parser = argparse.ArgumentParser(description="Trains and evaluates Gradient Boosted Trees model on MNIST")
    parser.add_argument("--train_images", type=str, default="./data/train-images-idx3-ubyte.gz", help="Train images in gz format described in http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("--train_labels", type=str, default="./data/train-labels-idx1-ubyte.gz", help="Train labels in gz format described in http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("--test_images", type=str, default="./data/t10k-images-idx3-ubyte.gz", help="Test images in gz format described in http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("--test_labels", type=str, default="./data/t10k-labels-idx1-ubyte.gz", help="Test labels in gz format described in http://yann.lecun.com/exdb/mnist/")
    parser.add_argument("--eval", type=str, choices=["valid", "test"], default="test", help="'valid' for evaluating on validation dataset, 'test' for evaluating on test dataset")
    parser.add_argument("--trees", type=int, default=200, help="Number of trees in the model")
    parser.add_argument("--learning_rate", type=float, default=0.25, help="Learning rate of the model")
    parser.add_argument("--num_leaves", type=int, default=32, help="Number of leaves for trees in the model")
    parser.add_argument("--output", type=str, choices=["all", "error_rate"], default="error_rate", help="'all' for printing more metrics, 'error_rate' for printing only error rate")
    parser.add_argument("--preprocessing", type=str, choices=["none", "deskew", "center", "augment", "augment+deskew", "center+augment", "center+deskew", "all"], 
                                default="deskew", help="Type of preprocessing used")

    args = parser.parse_args()
    if args.preprocessing == "all":
        args.preprocessing = "augment+deskew+center"

    return args


# Exctracts args from argparse module to dictionary
def get_model_params(args):
    return {"trees" : args.trees,
            "num_leaves" : args.num_leaves,
            "learning_rate" : args.learning_rate}


# Reads int from binary file with big endian
def read_int(ifile):
    return struct.unpack('>i', ifile.read(4))[0]       # big endian


# Loads images and labels file, returns them as numpy arrays 
def load_mnist_numpy(images_filename, labels_filename):
    with gzip.open(images_filename, "rb") as images_file:
        magic_number = read_int(images_file)
        if magic_number != 2051:
            raise Exception("MNIST images file does not have correct format")

        image_count = read_int(images_file)
        height = read_int(images_file)
        width = read_int(images_file)
        
        images = np.fromstring(images_file.read(), dtype=np.uint8)          # fromfile does not work with gzipped files
        images = images.reshape((image_count, -1)).astype(np.float32)

    with gzip.open(labels_filename, "rb") as labels_file:
        magic_number = read_int(labels_file)
        if magic_number != 2049:
            raise Exception("MNIST labels file does not have correct format")

        label_count = read_int(labels_file)
        if label_count != image_count:
            raise Exception("Images and labels file do not match")
        
        labels = np.fromstring(labels_file.read(), dtype=np.uint8)          # fromfile does not work with gzipped files

    return Dataset(images, labels, height, width)


# Splits dataset to two parts
def split_dataset(dataset, first_dataset_rows):
    first_images = dataset.images[0:first_dataset_rows]
    first_labels = dataset.labels[0:first_dataset_rows]
    
    second_images = dataset.images[first_dataset_rows:]
    second_labels = dataset.labels[first_dataset_rows:]

    h = dataset.height
    w = dataset.width

    return Dataset(first_images, first_labels, h, w), Dataset(second_images, second_labels, h, w)


# Shows MNIST image on the screen
def show(image):
    plt.imshow(image.reshape((28, 28)), cmap='gray')
    plt.show()

# Returns center of mass position, 0 is middle position
def get_center_1d(data):
    return int(np.mean(data * np.linspace(-data.shape[0] / 2, data.shape[0] / 2, data.shape[0])) / 256)


# Centers each image individually
def center_dataset(dataset):
    for i, image in enumerate(dataset.images):
        image = image.reshape((dataset.height, dataset.width))

        y_center = get_center_1d(np.sum(image, 1))
        x_center = get_center_1d(np.sum(image, 0))

        new_image = image[max(0, y_center):dataset.height + min(0, y_center),
                      max(0, x_center):dataset.width + min(0, x_center)]

        new_image = np.pad(new_image, ((max(0, -y_center), max(0, y_center)), (max(0, -x_center), max(0, x_center))), "constant")

        dataset.images[i] = new_image.reshape((dataset.height * dataset.width))


# Returns dataset with all original images together with some additional generated images
def augment_dataset(dataset):
    operations = []
    operations.append(Augmentor.Operations.Distort(probability=1.0, grid_width=8, grid_height=8, magnitude=1))

    new_images = []
    new_labels = []
    for i, (image, label) in enumerate(zip(dataset.images, dataset.labels)):
        image = image.reshape((dataset.height, dataset.width))

        pil_image = Image.fromarray(image)
        
        for operation in operations:
            pil_image = operation.perform_operation([pil_image])[0]

        new_image = np.array(pil_image.getdata())
        new_images.append(new_image)
        new_labels.append(label)
        
    all_images = np.vstack((dataset.images, np.array(new_images)))
    all_labels = np.hstack((dataset.labels, np.array(new_labels)))

    return Dataset(all_images, all_labels, dataset.height, dataset.width)


# from https://fsix.github.io/mnist/Deskewing.html
def moments(image):
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


# from https://fsix.github.io/mnist/Deskewing.html
def deskew(image):
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)


# Deskews (straighten) all images in dataset
def deskew_dataset(dataset):
    for i, image in enumerate(dataset.images):
        deskewed_image = deskew(image.reshape(dataset.height, dataset.width))
        dataset.images[i] = deskewed_image.reshape((dataset.height * dataset.width))


# Trains model, evaluates it and prints metrics
def train_and_eval_model(train, eval, model_params, output, preprocessing):
    if "center" in preprocessing:
        center_dataset(train)
        center_dataset(eval)

    if "deskew" in preprocessing:
        deskew_dataset(train)
        deskew_dataset(eval)

    if "augment" in preprocessing:
        train = augment_dataset(train)

    start = time.time()
    model = lightgbm.LGBMClassifier(n_estimators=model_params["trees"], 
                                    num_leaves=model_params["num_leaves"],
                                    learning_rate=model_params["learning_rate"],
                                    verbose=-1)
    model.fit(train.images, train.labels, verbose=(output == "all"))

    predictions = model.predict(eval.images)

    if output == "all":
        print("Time: {}".format(time.time() - start))
        print("Train error rate: {:.2f}%".format((model.predict(train.images) != train.labels).mean() * 100.0))

    print("Error rate: {:.2f}%".format((predictions != eval.labels).mean() * 100.0))

    return model


def main():
    args = parse_args()

    model_params = get_model_params(args)

    train = load_mnist_numpy(args.train_images, args.train_labels)
    test = load_mnist_numpy(args.test_images, args.test_labels)

    if args.eval == "valid":
        train, eval = split_dataset(train, 50000)
        train_and_eval_model(train, eval, model_params, args.output, args.preprocessing)
    else:
        train_and_eval_model(train, test, model_params, args.output, args.preprocessing)


if __name__ == "__main__":
    main()
