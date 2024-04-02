import numpy as np
import pytest
from src.data_procesing.normalizacion_data import NormalizationData

@pytest.fixture
def sample_data():
    return np.array([1, 2, 3, 4, 5])

def test_soft_max(sample_data):
    nd = NormalizationData(sample_data)
    result = nd.soft_max()
    expected_result = np.exp(sample_data) / np.sum(np.exp(sample_data))
    np.testing.assert_allclose(result, expected_result)

def test_standarization(sample_data):
    nd = NormalizationData(sample_data)
    result = nd.standarization()
    expected_result = (sample_data - np.mean(sample_data)) / np.std(sample_data)
    np.testing.assert_allclose(result, expected_result)

def test_range(sample_data):
    nd = NormalizationData(sample_data)
    result = nd.range()
    expected_result = (sample_data - np.min(sample_data)) / (np.max(sample_data) - np.min(sample_data))
    np.testing.assert_allclose(result, expected_result)
