import torch
from tissuemosaic.misc_utils.misc import get_percentile, SmartPca
import pytest
from sklearn.decomposition import PCA
import numpy

def test_get_percentile(capsys):
    """ Transform a tensor into its percentiles which are in [0.0, 1.0] """
    data = torch.zeros((2, 10))
    data[0] = torch.arange(10)
    data[1] = -torch.arange(10)
    q_true = torch.zeros_like(data)
    q_true[0] = torch.linspace(0.0, 1.0, 10)
    q_true[1] = torch.linspace(1.0, 0.0, 10)

    q = get_percentile(data, dim=-1)
    diff = (q - q_true).abs()
    assert torch.all(diff < 1E-4)

    data = torch.randn((2, 10))
    q_true = torch.zeros_like(data)
    mask_data0_le_data1 = data[0] <= data[1]
    q_true[0, mask_data0_le_data1] = 0.0
    q_true[1, mask_data0_le_data1] = 1.0
    q_true[0, ~mask_data0_le_data1] = 1.0
    q_true[1, ~mask_data0_le_data1] = 0.0
    q = get_percentile(data, dim=-2)
    diff = (q - q_true).abs()
    assert torch.all(diff < 1E-4)


@pytest.mark.parametrize("n, p", [[200, 30], [20, 300]])
def test_pca(n, p):
    q = 0.75
    data = torch.randn((n, p))
    pca = SmartPca(preprocess_strategy='raw')
    x1 = pca.fit_transform(data, n_components=q)
    nq = x1.shape[-1]
    assert (pca.explained_variance_ratio_[nq] > q)


@pytest.mark.parametrize("n, p", [[20, 200], [20, 3000]])
def test_pca_vs_scikit(n, p, capsys):
    """ Compare myPCA vs scikit-learn PCA"""
    # n = sample
    # p = features

    def _compute_corr(x, y):
        if isinstance(x, numpy.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, numpy.ndarray):
            y = torch.from_numpy(y)

        x_std, x_mu = torch.std_mean(x, dim=-2)
        y_std, y_mu = torch.std_mean(y, dim=-2)
        cov = torch.mean((x - x_mu) * (y - y_mu), dim=-2)
        corr = cov / (x_std * y_std)
        return corr

    k = 2  # true rank of the data
    if torch.cuda.is_available():
        U_true = torch.randn((n, k), device=torch.device('cuda'))
        S_true = torch.randn(k, device=torch.device('cuda')).exp()
        V_true = torch.randn((k, p), device=torch.device('cuda'))
    else:
        U_true = torch.randn((n, k), device=torch.device('cpu'))
        S_true = torch.randn(k, device=torch.device('cpu')).exp()
        V_true = torch.randn((k, p), device=torch.device('cpu'))

    color = torch.zeros(n)
    color[:int(0.25 * n)] = 0
    color[int(0.25 * n):int(0.5 * n)] = 1
    color[int(0.5 * n):int(0.75 * n)] = 2
    color[int(0.75 * n):] = 3

    dx, dy = 3, 5
    U_true[color == 0] += torch.tensor([dx, dy])
    U_true[color == 1] += torch.tensor([dx, -dy])
    U_true[color == 2] += torch.tensor([-dx, dy])
    U_true[color == 3] += torch.tensor([-dx, -dy])
    # plt.scatter(U_true[:, 0], U_true[:, 1], c=color)

    data = U_true @ torch.diag(S_true) @ V_true
    q = min(data.shape)

    # My PCA
    pca = SmartPca(preprocess_strategy='raw')
    x1 = pca.fit_transform(data, n_components=q)

    # Compare against PCA in scikit-learn
    pca_scikit = PCA(n_components=q, copy=False, whiten=False,
                     svd_solver='auto', tol=0.0,
                     iterated_power='auto', random_state=0)
    x2 = pca_scikit.fit_transform(data.detach().cpu().numpy())

    corr_abs_x = _compute_corr(x1, x2).abs()
    assert torch.all(corr_abs_x[:k]) > 0.95

    a = pca.explained_variance_.detach().cpu().numpy()
    b = pca_scikit.explained_variance_
    variance_error = numpy.abs(a-b) / b
    assert numpy.all(variance_error[:k] < 0.01)
