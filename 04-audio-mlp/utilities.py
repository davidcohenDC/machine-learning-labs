import time
import IPython
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay


def add_feature(dataset):
    feature_pow2 = np.square(dataset)   # Elevamento al quadrato
    # feature_sqrt2 = np.sqrt(dataset)  # Radice quadrata
    feature_sin = np.sin(dataset)  # Funzione seno
    feature_cos = np.cos(dataset)  # Funzione coseno
    feature = np.concatenate((dataset, feature_pow2), axis=1)
    print(feature.shape)
    return feature


def load_dataset():
    # Caricamento del dataset (train)
    train_dataset = np.load('DBs/AudioDigits_ML21_22/audio_ml2122_train.npy')
    # Caricamento del dataset (test)
    test_x = np.load('DBs/AudioDigits_ML21_22/audio_ml2122_test_no_labels.npy')
    # Conversione dei dati (train)
    digit_column = 173
    speaker_column = 174
    train_x = train_dataset[:, :-2].astype(np.float64)
    train_y = train_dataset[:, speaker_column].astype(int)
    # Conversione dei dati (test)
    test_x = test_x.astype(np.float64)
    # Aggiunta di feature
    feature_x = add_feature(train_x)
    # feature_y = add_feature(train_y)
    feature_test_x = add_feature(test_x)

    """print("feature_x"+str(feature_x.shape))
    print("feature_y"+str(train_y.shape))
    print("feature_test_x"+str(feature_test_x.shape))"""
    return feature_x, train_y, feature_test_x


def normalize_data(train_x, test_x):
    # Applica normalizzazione al train e al test
    scaler = StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    return train_x, test_x


def build_model(n_features):
    digit_output = 10
    speaker_output = 31

    # Procedura di creazione del modello
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=[n_features]),
        tf.keras.layers.Dense(150, kernel_initializer='lecun_normal', activation='selu',
                              kernel_regularizer=tf.keras.regularizers.l1_l2()),
        tf.keras.layers.Dense(150, kernel_initializer='lecun_normal', activation='selu',
                              kernel_regularizer=tf.keras.regularizers.l1_l2()),
        tf.keras.layers.Dense(digit_output, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)  # SGD(Stochastic Gradient Descent) / Adam / ecc...
    # optimizer = tf.keras.optimizers.Adam(0.005)

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


def neural_network(train_x, train_y, valid_x, valid_y):
    model = build_model(train_x.shape[1])
    print(train_x.shape[1])
    model.summary()

    n_epochs = 500
    minibatch_size = 256

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    start = time.time()

    history = model.fit(train_x, train_y, validation_data=(valid_x, valid_y),
                        epochs=n_epochs, batch_size=minibatch_size, shuffle=True,
                        verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots(report_every=10, dot_every=1)])

    print('\nTraining MLP completato in %.2f s.' % (time.time() - start))
    plot_history(history)

    return model


def plot_history(history):
    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=0)

    plotter.plot({'Basic': history}, metric="accuracy")
    plt.ylim([0, 1])
    plt.ylabel('Accuracy')
    plt.show()

    plotter.plot({'Basic': history}, metric="loss")
    plt.ylabel('Loss')
    plt.show()


def plot_accuracy_predictions(test_y, predictions):
    print('Accuratezza sul test set:', accuracy_score(test_y, predictions) * 100, '%')
    fig, ax = plt.subplots(figsize=(12, 12))
    ConfusionMatrixDisplay.from_predictions(test_y, predictions, ax=ax,
                                            display_labels=['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                                                            'seven', 'eight', 'nine'])


def save_predictions(predictions):
    np.savetxt('Es7Predictions.txt', predictions.astype(int), fmt='%i')
    print('Ok')
