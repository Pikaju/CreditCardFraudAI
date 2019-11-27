from src import dataset
from src.ensemble import FraudDetectionEnsemble
import numpy as np
import math
from tensorflow import keras


def main():
    data = dataset.load('res/creditcard.csv')

    # Logarithmically scale the amount column, as it can get very large. Avoid 0.0 inputs.
    data[:, -2] = np.log(data[:, -2] + 1.0)

    # Separate inputs and outputs.
    x, y = data[:, :-1], data[:, -1]

    # Create test split.
    split = data.shape[0] // 5
    x_train, y_train = x[:-split], y[:-split]
    x_test, y_test = x[-split:], y[-split:]

    model = FraudDetectionEnsemble(1)
    model.compile(optimizer=keras.optimizers.Adam(), loss="mean_squared_error", metrics=[keras.metrics.binary_accuracy])

    print('Training ensemble...')
    model.fit(x_train, y_train, batch_size=256, validation_split=0.2, epochs=32)

    print('Evaluating ensemble...')
    model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
