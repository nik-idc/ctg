import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import keras
import matplotlib.pyplot as plt
import cv2 as cv
import argparse
from keras import layers, utils, Sequential

cur_dir = pathlib.PureWindowsPath(os.getcwd()).as_posix() # Current dir
img_width, img_height = 224, 224 # Image dims

def parse_args():
    parser = argparse.ArgumentParser(
        prog='ctg',
        description='This programs trains an AI model to recognise '
        'book genre based on its cover. For more info visit '
        'https://paperswithcode.com/paper/judging-a-book-by-its-cover'
        '\nAll credit goes to the people behind that study and dataset.')

    parser.add_argument('--images_path', dest='images_path',
                        default=f'{cur_dir}/images-lite/images-main')
    parser.add_argument('--model', dest='model',
                        choices=['new', 'resnet'], default='new')
    parser.add_argument('--epochs', dest='epochs', default=5)

    args = parser.parse_args()

    images_path = args.images_path
    model_path = 'models/new/model.keras' if args.model == 'new' else 'models/resnet/model.keras'
    results_path = 'results/new/new-results.txt' if args.model == 'new' else 'results/resnet/resnet-results.txt'
    epochs = args.epochs
    
    return (images_path, model_path, results_path, epochs)

def fetch_data(images_path):
    # Fetch data
    print("Fetching data...")
    print("\tImporting training data...")
    train_ds = keras.utils.image_dataset_from_directory(
        images_path,
        validation_split=0.2,
        subset="training",
        labels="inferred",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print("\tTraining data imported")

    print("\tImporting testing data...")
    val_ds = keras.utils.image_dataset_from_directory(
        images_path,
        validation_split=0.2,
        subset="validation",
        labels="inferred",
        seed=123,
        image_size=(224, 224),
        batch_size=32)
    print("\tTesting data imported")
    print("Data fetched")

    class_names = train_ds.class_names

    return (train_ds, val_ds, class_names)

def create_model(class_names):
    # Create model
    print("Creating model...")
    num_classes = len(class_names)
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
    print("Model created")

    # Compile model
    print("Compiling model...")
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    print("Model compiled")

    return model

def train_model(model, train_ds, val_ds, model_path, epochs):
    # Train model
    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    print("Model trained")

    # Save model
    print("Saving model...")
    model.save(model_path)
    print("Model saved")

def load_model(model_path, images_path):
    print("Loading model...")
    model = keras.models.load_model(model_path)
    print("Model loaded")

    print("Loading class names...")
    all_items = os.listdir(images_path)
    class_names = [item for item in all_items if os.path.isdir(
        os.path.join(images_path, item))]
    print("Class names loaded")

    return (model, class_names)

def calc_accuracy(class_names, model):
    # Get genres
    test_images_path = f"{cur_dir}/images-lite/images-test"
    genres = os.listdir(test_images_path)

    # Correctness scores, a.k.a. scores for results independent of the confidence
    correctness_score = 0
    max_correctness_score = 0
    correctness_scores = []

    # Overall scores, i.e. taking confidence into account
    overall_score = 0
    max_overall_score = 0
    overall_scores = []
    
    print("Testing...")
    for genre in genres:
        # Genre path and images
        genre_path = f"{test_images_path}/{genre}"
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
        print(f"\tTesting {genre} genre images...")
        for image in images:
            image_path = f"{genre_path}/{image}"

            img = tf.keras.utils.load_img(
                image_path, target_size=(img_height, img_width)
            )
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create a batch

            predictions = model.predict(img_array, verbose=False)
            score = tf.nn.softmax(predictions[0])

            predicted_class = class_names[np.argmax(score)]
            confidence = np.max(score)

            genre_correctness_score = genre_correctness_score + \
                1 if predicted_class == genre else genre_correctness_score

            genre_overall_score = genre_overall_score + \
                1 * confidence if predicted_class == genre else genre_overall_score

        genre_correctness_score = (
            genre_correctness_score / max_genre_correctness_score) * 100
        print(f"\t\tGenre correctness score is: {genre_correctness_score:.2f}%")
        correctness_score += genre_correctness_score
        correctness_scores.append((genre, genre_correctness_score))

        genre_overall_score = (genre_overall_score /
                               max_genre_overall_score) * 100
        print(f"\t\tGenre overall score is: {genre_overall_score:.2f}%")
        overall_score += genre_overall_score
        overall_scores.append((genre, genre_overall_score))
        print(f"\tDone testing genre {genre} images")

    correctness_score = (correctness_score / max_correctness_score) * 100
    print(f"\tCorrectness score is: {correctness_score:.2f}%")

    overall_score = (overall_score / max_overall_score) * 100
    print(f"\tOverall score is: {overall_score:.2f}%\n")

    sorted_correctness = sorted(correctness_scores, key=lambda x: x[1])
    most_correct_genre, most_correct_genre_score = sorted_correctness[-1]
    print(
        f"\tMost correct genre: {most_correct_genre}, with score: {most_correct_genre_score:.2f}%")
    least_correct_genre, least_correct_genre_score = sorted_correctness[0]
    print(
        f"\tLeast correct genre: {least_correct_genre}, with score: {least_correct_genre_score:.2f}%")
    average_correctness = sum(
        x[1] for x in sorted_correctness) / len(sorted_correctness)
    print(f"\tAverage correctness score: {average_correctness:.2f}%")

    sorted_overall = sorted(overall_scores, key=lambda x: x[1])
    best_overall_genre, best_overall_genre_score = sorted_overall[-1]
    print(
        f"\tBest overall genre: {best_overall_genre}, with score: {best_overall_genre_score:.2f}%")
    worst_overall_genre, worst_overall_genre_score = sorted_overall[0]
    print(
        f"\tWorst overall genre: {worst_overall_genre}, with score: {worst_overall_genre_score:.2f}%")
    average_overall = sum(x[1] for x in sorted_overall) / len(sorted_overall)
    print(f"\tAverage overall score: {average_overall:.2f}%")
    
    print("Testing done")

    return (correctness_score,
            overall_score,
            (most_correct_genre, most_correct_genre_score),
            (least_correct_genre, least_correct_genre_score),
            average_correctness,
            (best_overall_genre, best_overall_genre_score),
            (worst_overall_genre, worst_overall_genre_score),
            average_overall)

def save_results(results, results_path):
    file_action = 'w' if os.path.isfile(results_path) else 'x'
    with open(results_path, file_action) as results_file:
        print(f"Writing results into {results_path}")
        results_file.write(f"\tCorrectness score is: {results[0]:.2f}%\n")
        results_file.write(f"\tOverall score is: {results[1]:.2f}%\n\n")

        results_file.write(
            f"\tMost correct genre: {results[2][0]}, with score: {results[2][1]:.2f}%\n")
        results_file.write(
            f"\tLeast correct genre: {results[3][0]}, with score: {results[3][1]:.2f}%\n")
        results_file.write(
            f"\tAverage correctness score is: {results[4]:.2f}%\n\n")

        results_file.write(
            f"\tBest overall genre: {results[5][0]}, with score: {results[5][1]:.2f}%\n")
        results_file.write(
            f"\tWorst overall genre: {results[6][0]}, with score: {results[6][1]:.2f}%\n")
        results_file.write(
            f"\tAverage overall score: {results[7]:.2f}%\n")
        print(f"Results saved")

if __name__ == "__main__":
    images_path, model_path, results_path, epochs = parse_args()

    class_names = []
    model = 0
    if os.path.isfile(model_path):
        model, class_names = load_model(model_path, images_path)
    else:
        train_ds, val_ds, class_names = fetch_data(images_path)
        model = create_model(class_names)
        train_model(model, train_ds, val_ds, model_path, epochs)

    results = calc_accuracy(class_names, model)
    save_results(results, results_path)
