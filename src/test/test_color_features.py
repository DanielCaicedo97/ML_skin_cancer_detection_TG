import numpy as np
import cv2

from src.extract_features.color_features import ColorFeatures

def generate_gaussian_image(mean_r, std_r, mean_g, std_g, mean_b, std_b, shape=(512, 512)):
    # Generar datos aleatorios con distribución normal para cada canal
    red_channel = np.random.normal(mean_r, std_r, size=shape)
    green_channel = np.random.normal(mean_g, std_g, size=shape)
    blue_channel = np.random.normal(mean_b, std_b, size=shape)

    # Asegurarse de que los valores estén en el rango adecuado (0 a 255)
    red_channel = np.clip(red_channel, 0, 255).astype(np.uint8)
    green_channel = np.clip(green_channel, 0, 255).astype(np.uint8)
    blue_channel = np.clip(blue_channel, 0, 255).astype(np.uint8)

    # Combinar los canales para crear la imagen RGB
    gaussian_image = cv2.merge([blue_channel, green_channel, red_channel])
    return gaussian_image

def test_mean_and_std():
    # Definir parámetros para los canales RGB
    mean_r, std_r = 150, 20
    mean_g, std_g = 100, 15
    mean_b, std_b = 200, 25

    # Generar la imagen gaussiana
    gaussian_image = generate_gaussian_image(mean_r, std_r, mean_g, std_g, mean_b, std_b)

    cf = ColorFeatures(gaussian_image)

    mean_r_calculated, mean_g_calculated, mean_b_calculated = cf._mean_rgb()
    std_r_calculated, std_g_calculated, std_b_calculated = cf._std_deviation_rgb()

    # Verificar que los cálculos sean correctos
    assert np.isclose(mean_r_calculated, mean_r, atol=1.0)
    assert np.isclose(std_r_calculated, std_r, atol=1.0)
    assert np.isclose(mean_g_calculated, mean_g, atol=1.0)
    assert np.isclose(std_g_calculated, std_g, atol=1.0)
    assert np.isclose(mean_b_calculated, mean_b, atol=1.0)
    assert np.isclose(std_b_calculated, std_b, atol=1.0)
