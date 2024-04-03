import numpy as np
import pytest
from src.data_procesing.gaussian_test_data import GaussianTest

@pytest.fixture
def sample_data():
    np.random.seed(42)  # Establecer la semilla para reproducibilidad
    return np.random.normal(loc=0, scale=1, size=1000)  # Datos de ejemplo

def test_lilliefors(sample_data):
    gt = GaussianTest(sample_data)
    result = gt.lilliefors()
    assert (result == True)

def test_Kolmogorov_Smirnov(sample_data):
    gt = GaussianTest(sample_data)
    result = gt.Kolmogorov_Smirnov()
    assert (result == True)

def test_anderson_darling(sample_data):
    gt = GaussianTest(sample_data)
    result = gt.anderson_darling()
    assert (result == True)


@pytest.fixture
def non_normal_data():
    return np.random.rand(1000)  # Generar datos aleatorios no gaussianos

def test_lilliefors_non_normal(non_normal_data):
    gt = GaussianTest(non_normal_data)
    result = gt.lilliefors()
    assert  result == False

def test_Kolmogorov_Smirnov_non_normal(non_normal_data):
    gt = GaussianTest(non_normal_data)
    result = gt.Kolmogorov_Smirnov()
    assert  result == False
    
def test_anderson_darling_non_normal(non_normal_data):
    gt = GaussianTest(non_normal_data)
    result = gt.anderson_darling()
    assert  result == False