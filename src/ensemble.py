from src.mlp import FraudDetectionMLP
from tensorflow import keras
import numpy as np


class FraudDetectionEnsemble:
    def __init__(self, size):
        self.models = [FraudDetectionMLP() for _ in range(size)]

    def compile(self, optimizer, loss, metrics):
        for model in self.models:
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def fit(self, x_train, y_train, validation_split, batch_size, epochs):
        for i, model in enumerate(self.models):
            print('Fitting model {}/{}...'.format(i + 1, len(self.models)))
            callbacks = [
                keras.callbacks.EarlyStopping(patience=2),
                keras.callbacks.ModelCheckpoint('res/ensemble_{}.hdf5'.format(i), save_best_only=True)
            ]
            model.fit(x_train, y_train, validation_split=validation_split, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def evaluate(self, x_test, y_test):
        y_preds = []
        for model in self.models:
            y_preds.append(np.round(model.predict(x_test)))
        y_preds = np.array(y_preds)
        ensemble_pred = np.squeeze(np.median(y_preds, axis=0))
        print('Ensemble accuracy:', np.mean(np.equal(ensemble_pred, y_test)))
