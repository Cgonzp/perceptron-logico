
import numpy as np
import pytest

from src.perceptron import Perceptron, hardlims


@pytest.fixture
def dataset():
    datos = np.array([
        [-1, -1, -1, -1],
        [-1, -1,  1, -1],
        [-1,  1, -1, -1],
        [-1,  1,  1,  1],
        [ 1, -1, -1, -1],
        [ 1, -1,  1,  1],
        [ 1,  1, -1,  1],
        [ 1,  1,  1,  1]
    ])

    return datos[:, :3], datos[:, 3]


def test_hardlims():
    assert hardlims(-2) == -1
    assert hardlims(0) == 1
    assert hardlims(2) == 1


def test_training_convergence(dataset):
    X, y = dataset

    model = Perceptron(random_state=0)
    model.fit(X, y)

    assert model.converged
    assert model.n_epochs <= model.max_epochs


def test_predictions(dataset):
    X, y = dataset

    model = Perceptron(random_state=0)
    model.fit(X, y)

    predictions = model.predict(X)

    assert np.array_equal(predictions, y)


def test_invalid_labels():
    X = np.array([[1, -1], [-1, 1]])
    y = np.array([0, 1])

    model = Perceptron()

    with pytest.raises(ValueError):
        model.fit(X, y)


def test_invalid_dimensions():
    X = np.array([[1, -1], [-1, 1]])
    y = np.array([1, -1])

    model = Perceptron()
    model.fit(X, y)

    with pytest.raises(ValueError):
        model.predict(np.array([[1, 1, 1]]))


def test_predict_before_training():
    model = Perceptron()

    with pytest.raises(RuntimeError):
        model.predict(np.array([[1, -1]]))

from src.visualization import (
    plot_training_history,
    plot_confusion_matrix
)


def test_confusion_matrix():
    y_true = np.array([-1, -1, 1, 1])
    y_pred = np.array([-1, 1, 1, 1])

    matrix = plot_confusion_matrix(y_true, y_pred, show=False)

    expected = np.array([
        [1, 1],
        [0, 2]
    ])

    assert np.array_equal(matrix, expected)


def test_training_history_without_training():
    model = Perceptron()

    with pytest.raises(ValueError):
        plot_training_history(model)
