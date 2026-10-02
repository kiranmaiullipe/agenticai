import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0

    mape = np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100

    return mae, rmse, mape
