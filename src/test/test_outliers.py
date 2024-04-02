import numpy as np
import pytest
from src.data_procesing.outliers_data import Outliers

@pytest.fixture
def sample_data():
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])  # Datos de ejemplo

def test_remove_outliers_std(sample_data):
    ol = Outliers(sample_data)
    result = ol.remove_outliers_std()
    expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Resultado esperado después de eliminar outliers
    np.testing.assert_array_equal(result, expected_result)

def test_remove_outliers_iqr(sample_data):
    ol = Outliers(sample_data)
    result = ol.remove_outliers_iqr()
    expected_result = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Resultado esperado después de eliminar outliers
    np.testing.assert_array_equal(result, expected_result)
