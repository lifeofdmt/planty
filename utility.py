import argparse
import json
import torch
from PIL import Image
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    return ax


def load_cat(file):
    """
    Load dictionary of category to name mapping from file
    """
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def program_parser(train=True):
    """
    Returns parser for command line arguments
    setting `train=True` returns parser for training args
    and setting `train=False` returns parser for prediction args
    """
    # Create command line parser
    parser = argparse.ArgumentParser(prog="train.py", description="Trains plant image classifier")
    
    if train:
        parser.add_argument('data_dir')
        parser.add_argument('--save_dir')
        parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'resnet50'])
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--hidden_units', default=3, type=int)
    else:
        parser.add_argument('input')
        parser.add_argument('checkpoint')
        parser.add_argument('--top_k', default=5, type=int)
        parser.add_argument('--category_names')
 
    parser.add_argument('--gpu', default=False, action="store_const", const=True)
    return parser

def configure_device(gpu=False):
    """
    Configures device training and prediction device
    """
    cuda_available = torch.cuda.is_available()
    device = 'cpu'
    if gpu:
        if cuda_available:
            device = 'cuda'
    return torch.device(device)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image, mode="r")
    image.thumbnail((256, 256))

    # Crop center of image
    width, height = image.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    image.show()

    np_image = np.array(image)
    np_image = np_image / 255 # Normalize color channels

    # Transpose image dimensions
    np_image = np.transpose(np_image, (2, 0, 1))

    # Normalize image
    means = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))

    np_image = (np_image - means) / std
    return np_image

