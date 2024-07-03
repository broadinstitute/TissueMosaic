import pytest
import torch
import numpy
from tissuemosaic.model_utils.mlp_classifier_regressor import PlRegressor, PlClassifier


@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_regressor(x_type, y_type):
    max_iter = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randn(n)
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    regressor = PlRegressor(max_iter=max_iter)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    score = regressor.score(X, y)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert y_pred.shape[0] == n
    loss = regressor.loss_
    loss_curve = regressor.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter


@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_classifier(x_type, y_type):
    max_iter = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randint(low=3, high=5, size=[n])
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    classifier = PlClassifier(max_iter=max_iter)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    score = classifier.score(X, y)
    prob = classifier.predict_proba(X)
    log_prob = classifier.predict_log_proba(X)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert isinstance(prob, numpy.ndarray)
    assert isinstance(log_prob, numpy.ndarray)
    assert prob.shape == log_prob.shape
    assert y_pred.shape[0] == n

    loss = classifier.loss_
    loss_curve = classifier.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter


@pytest.mark.parametrize("hard_bootstrapping", [True, False])
@pytest.mark.parametrize("x_type", ["torch", "numpy"])
@pytest.mark.parametrize("y_type", ["torch", "numpy", "list"])
def test_noisy_classifier(x_type, y_type, hard_bootstrapping):
    max_epochs = 5
    n, d = 10, 3
    X = torch.randn(n, d)
    y = torch.randint(low=3, high=5, size=[n])
    if x_type == "numpy":
        X = X.cpu().numpy()
    if y_type == "numpy":
        y = y.cpu().numpy()
    elif y_type == "list":
        y = y.cpu().tolist()

    classifier = PlClassifier(
        max_epochs=max_epochs,
        noisy_labels=True,
        bootstrap_epoch_start=3,
        lambda_reg=1.0,
        hard_bootstrapping=hard_bootstrapping)

    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    score = classifier.score(X, y)
    prob = classifier.predict_proba(X)
    log_prob = classifier.predict_log_proba(X)
    assert isinstance(score, float)
    assert isinstance(y_pred, numpy.ndarray)
    assert isinstance(prob, numpy.ndarray)
    assert isinstance(log_prob, numpy.ndarray)
    assert prob.shape == log_prob.shape
    assert y_pred.shape[0] == n

    loss = classifier.loss_
    loss_curve = classifier.loss_curve_
    assert isinstance(loss, float)
    assert isinstance(loss_curve, list) and len(loss_curve) == max_iter
