import cv2
import pytest
from src.extract_features.color_features import ColorFeatures

@pytest.fixture
def sample_image():
    # Creamos una imagen de ejemplo para las pruebas
    img = cv2.imread('src/data_example/raw/ISIC_1435135.JPG')
    return img

def test_color_features_mean(sample_image):
    cf = ColorFeatures(sample_image)
    mean = cf._mean_rgb()
    assert mean['mean_r'] == pytest.approx(127.5, abs=1)
    assert mean['mean_g'] == pytest.approx(127.5, abs=1)
    assert mean['mean_b'] == pytest.approx(127.5, abs=1)

def test_color_features_variance(sample_image):
    cf = ColorFeatures(sample_image)
    variance = cf._variance_rgb()
    assert variance['variance_r'] == pytest.approx(0, abs=1)
    assert variance['variance_g'] == pytest.approx(0, abs=1)
    assert variance['variance_b'] == pytest.approx(0, abs=1)

def test_color_features_std_deviation(sample_image):
    cf = ColorFeatures(sample_image)
    std_deviation = cf._std_deviation_rgb()
    assert std_deviation['std_deviation_r'] == pytest.approx(0, abs=1)
    assert std_deviation['std_deviation_g'] == pytest.approx(0, abs=1)
    assert std_deviation['std_deviation_b'] == pytest.approx(0, abs=1)
