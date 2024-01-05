import numpy as np
import os
import PIL
import PIL.Image
from sklearn.metrics import auc
import tensorflow as tf
import pathlib
import keras
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
from keras import layers, utils, Sequential, applications, optimizers, losses

cur_dir = pathlib.PureWindowsPath(os.getcwd()).as_posix()  # Current dir
img_width, img_height = 224, 224  # Image dims

MODEL_NAMES = [
    'new',
    'resnet50',
    'mobile_net',
    'mobile_net_v2',
    'dense_net_121',
    'nas_net_mobile',
]

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

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ctg',
        description='This programs trains an AI model to recognise '
        'book genre based on its cover. For more info visit '
        'https://paperswithcode.com/paper/judging-a-book-by-its-cover'
        '\nAll credit goes to the people behind that study and dataset.')

    parser.add_argument('-tr', '--train_images_path',
                        default=f'{cur_dir}/images/images-train')
    parser.add_argument('-te', '--test_images_path',
                        default=f'{cur_dir}/images/images-test')
    parser.add_argument('-a', '--use_all', action='store_true', default=False)
    parser.add_argument('-n', '--model_names',
                        choices=MODEL_NAMES,
                        action='append')
    parser.add_argument('-e', '--epochs', dest='epochs', default=5)

    args = parser.parse_args()

    train_images_path = args.train_images_path
    test_images_path = args.test_images_path
    use_all = args.use_all
    model_names = args.model_names
    epochs = args.epochs

    return (train_images_path, test_images_path, use_all, model_names, epochs)


def fetch_data(images_path):
    # Fetch data
    print('Fetching data...')
    print('\tImporting training data...')
    train_ds = keras.utils.image_dataset_from_directory(
        images_path,
        validation_split=0.2,
        subset='training',
        labels='inferred',
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print('\tTraining data imported')

    print('\tImporting testing data...')
    val_ds = keras.utils.image_dataset_from_directory(
        images_path,
        validation_split=0.2,
        subset='validation',
        labels='inferred',
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print('\tTesting data imported')
    print('Data fetched')

    # class_names = train_ds.class_names

    return (train_ds, val_ds)


def create_new_model():
    # Create model
    print('Creating new model...')
    num_classes = len(CLASS_NAMES)
    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])
    print('Model created')

    # Compile model
    print('Compiling model...')
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    print('Model compiled')

    return model


def gen_pretrained_model(model_name):
    print('Generating pretrained model...')
    base_model = ...
    if model_name == 'resnet50':
        print('\tGenerating Resnet50')
        base_model = applications.ResNet50(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(img_width, img_height, 3),
            include_top=False)
    elif model_name == 'mobile_net':
        print('\tGenerating MobileNet')
        base_model = applications.MobileNet(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(img_width, img_height, 3),
            include_top=False)
    elif model_name == 'mobile_net_v2':
        print('\tGenerating MobileNetV2')
        base_model = applications.MobileNetV2(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(img_width, img_height, 3),
            include_top=False)
    elif model_name == 'dense_net_121':
        print('\tGenerating DenseNet121')
        base_model = applications.DenseNet121(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(img_width, img_height, 3),
            include_top=False)
    elif model_name == 'nas_net_mobile':
        print('\tGenerating NASNetMobile')
        base_model = applications.NASNetMobile(
            weights='imagenet',  # Load weights pre-trained on ImageNet.
            input_shape=(img_width, img_height, 3),
            include_top=False)

    # Freeze the model
    print('\tFreezing base model')
    base_model.trainable = False

    print('\tCreating model inputs')
    inputs = keras.Input(shape=(img_width, img_height, 3))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    print('\tCreating model outputs')
    outputs = keras.layers.Dense(len(CLASS_NAMES))(x)

    print('\tCreating keras model')
    model = keras.Model(inputs, outputs)

    # Compile the model
    print('\tCompiling model')
    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    
    print('Done generating pretrained model')
    
    return model


def train_model(model, model_name, model_path, train_ds, val_ds, epochs):
    # Train model
    print(f'Training model "{model_name}" ...')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    print('Model trained')

    # Save model
    print('Saving model...')
    model.save(model_path)
    print('Model saved')


def load_model(model_path):
    print('Loading model...')
    model = keras.models.load_model(model_path)
    print('Model loaded')

    return model


def calc_accuracy(test_images_path, model):
    # Get genres
    genres = os.listdir(test_images_path)

    # Correctness scores, a.k.a. scores for results independent of the confidence
    correctness_score = 0
    max_correctness_score = 0
    correctness_scores = []

    # Overall scores, i.e. taking confidence into account
    overall_score = 0
    max_overall_score = 0
    overall_scores = []

    print('Testing...')
    for genre in genres:
        # Genre path and images
        genre_path = f'{test_images_path}/{genre}'
        images = [f for f in os.listdir(genre_path) if os.path.isfile(
            os.path.join(genre_path, f))]

        # Correctness core vars
        genre_correctness_score = 0
        max_genre_correctness_score = len(images)
        max_correctness_score += max_genre_correctness_score

        # Overall core vars
        genre_overall_score = 0
        max_genre_overall_score = len(images)
        max_overall_score += max_genre_overall_score

        # Iterate and test every image in the genre
        print(f'\tTesting {genre} genre images...')
        for image in images:
            image_path = f'{genre_path}/{image}'

            img = tf.keras.utils.load_img(
                image_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array, verbose=False)
            score = tf.nn.softmax(predictions[0])

            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = np.max(score)

            genre_correctness_score = genre_correctness_score + \
                1 if predicted_class == genre else genre_correctness_score

            genre_overall_score = genre_overall_score + \
                1 * confidence if predicted_class == genre else genre_overall_score

        genre_correctness_score = (
            genre_correctness_score / max_genre_correctness_score) * 100
        print(
            f'\t\tGenre correctness score is: {genre_correctness_score:.2f}%')
        correctness_score += genre_correctness_score
        correctness_scores.append((genre, genre_correctness_score))

        genre_overall_score = (genre_overall_score /
                               max_genre_overall_score) * 100
        print(f'\t\tGenre overall score is: {genre_overall_score:.2f}%')
        overall_score += genre_overall_score
        overall_scores.append((genre, genre_overall_score))
        print(f'\tDone testing genre {genre} images')

    correctness_score = (correctness_score / max_correctness_score) * 100
    print(f'\tCorrectness score is: {correctness_score:.2f}%')

    overall_score = (overall_score / max_overall_score) * 100
    print(f'\tOverall score is: {overall_score:.2f}%\n')

    sorted_correctness = sorted(correctness_scores, key=lambda x: x[1])
    most_correct_genre, most_correct_genre_score = sorted_correctness[-1]
    print(
        f'\tMost correct genre: {most_correct_genre}, with score: {most_correct_genre_score:.2f}%')
    least_correct_genre, least_correct_genre_score = sorted_correctness[0]
    print(
        f'\tLeast correct genre: {least_correct_genre}, with score: {least_correct_genre_score:.2f}%')
    average_correctness = sum(
        x[1] for x in sorted_correctness) / len(sorted_correctness)
    print(f'\tAverage correctness score: {average_correctness:.2f}%')

    sorted_overall = sorted(overall_scores, key=lambda x: x[1])
    best_overall_genre, best_overall_genre_score = sorted_overall[-1]
    print(
        f'\tBest overall genre: {best_overall_genre}, with score: {best_overall_genre_score:.2f}%')
    worst_overall_genre, worst_overall_genre_score = sorted_overall[0]
    print(
        f'\tWorst overall genre: {worst_overall_genre}, with score: {worst_overall_genre_score:.2f}%')
    average_overall = sum(x[1] for x in sorted_overall) / len(sorted_overall)
    print(f'\tAverage overall score: {average_overall:.2f}%')

    print('Testing done')

    return (correctness_score,
            overall_score,
            (most_correct_genre, most_correct_genre_score),
            (least_correct_genre, least_correct_genre_score),
            average_correctness,
            (best_overall_genre, best_overall_genre_score),
            (worst_overall_genre, worst_overall_genre_score),
            average_overall)


def save_results(results, results_path):
    file_action = 'w' if os.path.exists(results_path) else 'x'
    with open(results_path, file_action) as results_file:
        print(f'Writing results into {results_path}')
        results_file.write(f'Correctness score is: {results[0]:.2f}%\n')
        results_file.write(f'Overall score is: {results[1]:.2f}%\n\n')

        results_file.write(
            f'Most correct genre: {results[2][0]}, with score: {results[2][1]:.2f}%\n')
        results_file.write(
            f'Least correct genre: {results[3][0]}, with score: {results[3][1]:.2f}%\n')
        results_file.write(
            f'Average correctness score is: {results[4]:.2f}%\n\n')

        results_file.write(
            f'Best overall genre: {results[5][0]}, with score: {results[5][1]:.2f}%\n')
        results_file.write(
            f'Worst overall genre: {results[6][0]}, with score: {results[6][1]:.2f}%\n')
        results_file.write(
            f'Average overall score: {results[7]:.2f}%\n')
        print(f'Results saved')


if __name__ == '__main__':
    # Parse args
    train_images_path, test_images_path, use_all, model_names, epochs = parse_args()
    model_names = MODEL_NAMES if use_all else model_names
        
    # Create model path and results arrays
    print(f'Testing {len(model_names)} models:\n\t{model_names}')
    print('Creating model and results paths')
    model_paths, results_paths = [], []
    for model_name in model_names:
        # Make directories if they do not exists
        model_folder = f'{cur_dir}/models/{model_name}'
        model_results_folder = f'{cur_dir}/results/{model_name}'
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(model_results_folder, exist_ok=True)
        
        # Create model and its results paths and append them
        model_path = f'{model_folder}/{model_name}-model.keras'
        results_path = f'{model_results_folder}/{model_name}-results.txt'
        model_paths.append(model_path)
        results_paths.append(results_path)
        
        print(f'\tModel path: {model_paths[-1]}')
        print(f'\tModel results path: {results_paths[-1]}')
    print('Model and results paths created')

    # Train (if needed) and test supplied models
    for i in range(len(model_names)):
        model = 0
        # Load model if trained, train if not trained
        if os.path.exists(model_paths[i]): # If model is already trained
            print(f'Model "{model_names[i]}" is already trained, loading it')
            model = load_model(model_paths[i])
            print(f'Model "{model_names[i]}" loaded')
        else: # If model is not yet trained
            print(f'Model "{model_names[i]}" is not yet trained, beginning training')
            train_ds, val_ds = fetch_data(train_images_path)
            model = ...
            if model_names[i] == 'new':
                model = create_new_model()
            else:
                model = gen_pretrained_model(model_names[i])
            train_model(model, model_names[i], model_paths[i], train_ds, val_ds, epochs)
            print(f'Model "{model_names[i]}" trained')

        # Get results if not yet tested
        if not os.path.exists(results_paths[i]):
            results = calc_accuracy(test_images_path, model)
            save_results(results, results_paths[i])
        else:
            print(f'Results for model "{model_names[i]}" already exist in '
                  f'the following location: "{results_paths[i]}"')
