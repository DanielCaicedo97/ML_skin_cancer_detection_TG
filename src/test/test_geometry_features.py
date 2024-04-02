import numpy as np
import cv2 
import pytest
from src.extract_features.geometry_features import GeometryFeatures

@pytest.fixture
def binary_image_square():
    # Crear una imagen binaria de 512x512 con un cuadrado
    image = np.zeros((512, 512), dtype=np.uint8)

    # Definir las coordenadas del cuadrado
    x1, y1 = (512 - 100) // 2, (512 - 100) // 2  # Esquina superior izquierda
    x2, y2 = x1 + 100, y1 + 100  # Esquina inferior derecha

    # Dibujar el cuadrado en la imagen
    image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)  # -1 para dibujar el cuadrado relleno
    return image

def test_compactness_index_square(binary_image_square):
    # Lado del cuadrado 
    L = 100
    expected_compactness_index  = ((4*L)**2)/(4*np.pi*L**2)
    #calcular el indice de compacidad dada una imagen
    gf = GeometryFeatures(binary_image_square)
    compactness_index = gf._compactness_index()
    assert compactness_index == expected_compactness_index

@pytest.fixture
def binary_ellipse_image():
    # Crear una imagen binaria de 512x512 con una elipse descentrada
    image = np.zeros((512, 512), dtype=np.uint8)
    center_coordinates = (200, 300)
    axes_length = (200, 100)
    angle = 30  # Ángulo de inclinación de la elipse
    color = 255  # Blanco
    thickness = -1  # Relleno
    image = cv2.ellipse(image, center_coordinates, axes_length, angle, 0, 360, color, thickness)
    cv2.imshow('elipse', image)
    return image

def test_symmetry(binary_ellipse_image):
    # Calcular la simetría de la imagen binaria del círculo
    gf = GeometryFeatures(binary_ellipse_image)
    symmetry_values = gf._symmetry()
    
    # Comprobar que los valores de simetría están en el rango [0, 1]
    for value in symmetry_values:
        assert 0 <= value <= 1

    # Comprobar que la simetría horizontal y vertical es aproximadamente 1 (perfecta simetría)
    assert np.isclose(symmetry_values[0], 1, atol=0.15)  # Se permite un error del 15%
    assert np.isclose(symmetry_values[1], 1, atol=0.15)  # Se permite un error del 15%