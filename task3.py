import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os
from keras.preprocessing import image
from keras.utils import image_dataset_from_directory

if not 'history' in st.session_state:
    st.session_state.history = None

# Set page title and background color
st.set_page_config(
    page_title="Salih Ekici's DL Task",
    page_icon="✏️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Salih Ekici's DL task")

if st.sidebar.button("EDA"):
    st.header("Data analysis")
    st.write("In this deep learning task, I decided to train a model that would be able to differentiate between 5 types of animals: dogs, cats, kangaroos, pandas and sharks.")
    st.write("The reason that I chose these animals was because I really like animals and making an image classifier that could predict a type of animal sounded really cool to me.")
    st.write("Let's first of all check how many training images we will be working with")
    training_directory = "./datasets/animals/training"
    classes = os.listdir(training_directory)
    for class_name in classes:
        class_path = os.path.join(training_directory, class_name)
        num_images = len(os.listdir(class_path))
        st.write(f"{class_name} class, Number of Images: {num_images}")

# Set the parameters for your data
batch_size = 32
image_size = (224, 224)
validation_split = 0.2

train_ds = image_dataset_from_directory(
    directory='./datasets/animals/training',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='training',
    seed=123
)

# Create the validation dataset from the 'train' directory
validation_ds = image_dataset_from_directory(
    directory='./datasets/animals/training',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size,
    validation_split=validation_split,
    subset='validation',
    seed=123
)

# Create the testing dataset from the 'test' directory
test_ds = image_dataset_from_directory(
    directory='./datasets/animals/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=image_size
)

epoch_slider = st.sidebar.slider("Number of epochs", min_value=5, max_value=30, value=15, step=5)
batch_size_dropdown = st.sidebar.selectbox("Select the batch size",("32","64","128"))


# Choose the number of classes that you will be working with
NUM_CLASSES = 5
# choose the image size
IMG_SIZE = 128
# There is no shearing option anymore, but there is a translation option
HEIGTH_FACTOR = 0.2
WIDTH_FACTOR = 0.2

# Create a sequential model with a list of layers
model = tf.keras.Sequential([
# Add a resizing layer to resize the images to a consistent shape
layers.Resizing(IMG_SIZE, IMG_SIZE),
# Add a rescaling layer to normalize the values of the pixels
layers.Rescaling(1./255),
# Add some data augmentation layers to apply random transformations during training
layers.RandomFlip("horizontal"),
layers.RandomTranslation(HEIGTH_FACTOR,WIDTH_FACTOR),
layers.RandomZoom(0.2),
# add a conv2d filter that will go over the image from left to right calculating filter maps
layers.Conv2D(32, (3, 3), input_shape = (IMG_SIZE, IMG_SIZE, 3), activation="relu"),
layers.MaxPooling2D((2, 2)),
# turn off 20% of the nodes in a random order so you dont overfit the model
layers.Dropout(0.2),
layers.Conv2D(32, (3, 3), activation="relu"),

layers.MaxPooling2D((2, 2)),
layers.Dropout(0.2),
layers.Flatten(),
layers.Dense(128, activation="relu"),
layers.Dense(NUM_CLASSES, activation="softmax")
])

# Compile and train your model as usual
# Compile and train your model as usual
model.compile(optimizer = optimizers.Adam(learning_rate=0.0001), 
            loss = 'categorical_crossentropy', 
            metrics = ['accuracy'])


if st.sidebar.button("Train the model"):
    # Modify hyperparameters
    new_epochs = epoch_slider
    new_batch_size = batch_size_dropdown

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    st.session_state.history = model.fit(train_ds, validation_data=validation_ds, epochs=int(new_epochs), batch_size=int(new_batch_size))
    st.write("Model has been trained, press the plot button on the sidebar to see the graphs")

if st.sidebar.button("Plot"):
    if st.session_state.history is not None:
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the loss curves on the first subplot
        ax1.plot(st.session_state.history.history['loss'], label='training loss')
        ax1.plot(st.session_state.history.history['val_loss'], label='validation loss')
        ax1.set_title('Loss curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Plot the accuracy curves on the second subplot
        ax2.plot(st.session_state.history.history['accuracy'], label='training accuracy')
        ax2.plot(st.session_state.history.history['val_accuracy'], label='validation accuracy')
        ax2.set_title('Accuracy curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Adjust the spacing between subplots
        fig.tight_layout()

        # Show the figure
        st.pyplot(fig)

        # Use the model to evaluate the test dataset
        test_loss, test_acc = model.evaluate(test_ds)
        st.write('Test accuracy (model):', test_acc)
