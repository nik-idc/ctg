from itertools import cycle
import numpy as np
import os
import PIL
import PIL.Image
from sklearn.calibration import label_binarize
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pathlib
import keras
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
from keras import layers, utils, Sequential, applications, optimizers, losses
from main import load_model

CUR_DIR = pathlib.PureWindowsPath(os.getcwd()).as_posix()
IMAGES_PATH = f'{CUR_DIR}/images/images-test'
ROC_PATH = f'{CUR_DIR}/plots'

IMG_WIDTH, IMG_HEIGHT = 224, 224

BATCH_SIZE = 32

MODEL_NAMES = [
    'new',
    'mobile_net',
    'mobile_net_v2',
    'dense_net_121',
    'nas_net_mobile',
    'resnet50',
]
NUM_MODELS = len(MODEL_NAMES)

CLASS_NAMES = [
    'Arts & Photography',
    'Business & Money',
    'Calendars',
    'Comics & Graphic Novels',
    'Computers & Technology',
    'Cookbooks, Food & Wine',
    'Engineering & Transportation',
    'History',
    'Literature & Fiction',
    'Medical Books',
    'Mystery, Thriller & Suspense',
    'Religion & Spirituality',
    'Romance',
    'Science & Math',
    'Science Fiction & Fantasy',
    'Teen & Young Adult',
]
NUM_CLASSES = len(CLASS_NAMES)

COLORS = [
    'aqua',
    'darkorange',
    'cornflowerblue',
    'limegreen',
    'tomato',
    'purple',
    'gold',
    'cyan',
    'magenta',
    'lightgreen',
    'lightblue',
    'salmon',
    'indigo',
    'red',
    'teal',
    'pink'
]
NUM_COLORS = len(COLORS)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ctg',
        description='This programs plots a ROC curve for specified model and classes. '
        'of cover-to-image AI. For more info visit '
        'https://paperswithcode.com/paper/judging-a-book-by-its-cover'
        '\nAll credit goes to the people behind that study and dataset.')

    parser.add_argument('-a', '--use_all', action='store_true', default=False)
    parser.add_argument('-n', '--model_names',
                        choices=MODEL_NAMES,
                        action='append')

    args = parser.parse_args()

    use_all = args.use_all
    model_names = args.model_names

    return (use_all, model_names)

def load_data(images_path: str):
    # Create a dataset from the directory
    # Change as per your image size requirements
    print(f'Generating ROC curve...')
    print(f'\tCreating dataset from directory')
    image_size = (IMG_WIDTH, IMG_HEIGHT)
    dataset = utils.image_dataset_from_directory(
        images_path,
        shuffle=False,
        image_size=image_size,
        batch_size=BATCH_SIZE
    )

    # Get labels for the test dataset
    print(f'\tCreating "X" and "y" arrays')
    X, loaded_y = [], []
    for index, (images, labels) in enumerate(dataset):
        print(f'\t\tBatch #{index}')
        X.append(images.numpy())
        loaded_y.append(labels.numpy())

    # Binarize the output
    print(f'\tBinarizing the output')
    y = []
    for index, y_batch in enumerate(loaded_y):
        print(f'\t\tBatch #{index}')
        binarized_batch = label_binarize(
            y_batch, classes=range(len(CLASS_NAMES)))
        y.append(binarized_batch)
        
    return (X, y)

def compute_scores(model: Sequential, X: list):
    print(f'\tComputing scores')
    y_scores = []
    for index, batch_X in enumerate(X):
        print(f'\t\tBatch #{index}')
        y_scores.append(model.predict(
            batch_X, batch_size=BATCH_SIZE, verbose=False))
        
    return y_scores

def unbatch_data(y: list, y_scores: list):
    print(f'\tUnbatching data')
    unbatched_y = []
    unbatched_y_scores = []
    for (y_batch, y_scores_batch) in zip(y, y_scores):
        unbatched_y.extend(y_batch)
        unbatched_y_scores.extend(y_scores_batch)
    unbatched_y = np.array(unbatched_y)
    unbatched_y_scores = np.array(unbatched_y_scores)
    
    return (unbatched_y, unbatched_y_scores)
    
def compute_roc(y: list, y_scores: list):
    print(f'\tComputing FPR, TPR and ROC AUC')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(CLASS_NAMES)):
        fpr[i], tpr[i], _ = roc_curve(
            y[:, i], y_scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    return (fpr, tpr, roc_auc)

def plot_roc(fpr, tpr, roc_auc, plot_path):
    # Plot all ROC curves
    print(f'\tPlotting')
    plt.figure(1)
    for i, color in zip(range(NUM_CLASSES), COLORS):
        print(f'\t\tPlotting ROC for class #{i}')
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    
    print(f'\tSaving ROC curve as a png file')
    plt.savefig(plot_path)
    
    plt.clf()
    
    # print(f'\tDisplaying plot')
    # plt.show()

def roc(model: Sequential, images_path: str, plot_path: str):
    print(f'Generating ROC curve...')
    X, y = load_data(images_path)
    y_scores = compute_scores(model, X)
    y, y_scores = unbatch_data(y, y_scores)
    fpr, trp, roc_auc = compute_roc(y, y_scores)    
    plot_roc(fpr, trp, roc_auc, plot_path)
    print(f'Done generating ROC curves')

def create_paths():
    roc_paths, model_paths = [], []
    for model_name in model_names:
        # Make directories if they do not exists
        model_folder = f'{CUR_DIR}/models/{model_name}'
        roc_folder = f'{CUR_DIR}/roc-plots/{model_name}'
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(roc_folder, exist_ok=True)

        # Create model and its ROC paths and append them
        model_path = f'{model_folder}/{model_name}-model.keras'
        roc_path = f'{roc_folder}/{model_name}-roc.png'
        model_paths.append(model_path)
        roc_paths.append(roc_path)

        print(f'\t"{model_name}" model path: {model_paths[-1]}')
        print(f'\t"{model_name}" model ROC path: {roc_paths[-1]}')
    print('Paths created')
    
    return (roc_paths, model_paths)

if __name__ == '__main__':
    # Parse args
    use_all, model_names = parse_args()
    model_names = MODEL_NAMES if use_all else model_names

    # Create model path and results arrays
    roc_paths, model_paths = create_paths()

    # Generate ROC curve for supplied models
    for model_name, model_path, roc_path in zip(model_names, model_paths, roc_paths):
        model = load_model(model_path)
        roc(model, IMAGES_PATH, roc_path)
