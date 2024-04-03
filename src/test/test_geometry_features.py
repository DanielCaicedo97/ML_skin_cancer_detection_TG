import cv2 
import pytest
import numpy as np
from src.extract_features.geometry_features import GeometryFeatures

@pytest.fixture
def binary_image_square():
    L = 200
    # Crear una imagen binaria de 512x512 con un cuadrado
    image = np.zeros((512, 512), dtype=np.uint8)

    # Definir las coordenadas del cuadrado
    x1, y1 = (512 - L) // 2, (512 - L) // 2  # Esquina superior izquierda
    x2, y2 = x1 + L, y1 + L  # Esquina inferior derecha

    # Dibujar el cuadrado en la imagen
    image = cv2.rectangle(image, (x1, y1), (x2, y2), 255, -1)  # -1 para dibujar el cuadrado relleno
    return image

@pytest.fixture
def binary_ellipse_image():
    # Crear una imagen binaria de 512x512 con una elipse descentrada
    image = np.zeros((512, 512), dtype=np.uint8)
    center_coordinates = (300, 300)
    axes_length = (200, 100)
    angle = 30  # Ángulo de inclinación de la elipse
    color = 255  # Blanco
    thickness = -1  # Relleno
    image = cv2.ellipse(image, center_coordinates, axes_length, angle, 0, 360, color, thickness)
    return image

def test_compactness_index_square(binary_image_square):
    # Lado del cuadrado 
    L = 200
    expected_compactness_index  = ((4*L)**2)/(4*np.pi*L**2)
    #calcular el indice de compacidad dada una imagen
    gf = GeometryFeatures(binary_image_square)
    compactness_index = gf._compactness_index()
    assert compactness_index == expected_compactness_index

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

def test_fractal_dimension_square(binary_image_square):
    # Calcular la dimensión fractal de la imagen binaria
    gf = GeometryFeatures(binary_image_square)
    fractal_dim = gf._fractal_dimension()
    # La dimensión fractal de una imagen completamente negra (sin píxeles blancos) es 0
    assert fractal_dim == pytest.approx(1.0, abs=5e-2)  # Permitimos un pequeño margen de error

def test_fractal_dimension_elipse(binary_ellipse_image):
    # Calcular la dimensión fractal de la imagen binaria
    gf = GeometryFeatures(binary_ellipse_image)
    fractal_dim = gf._fractal_dimension()
    # La dimensión fractal de una imagen completamente negra (sin píxeles blancos) es 0
    assert fractal_dim == pytest.approx(1.0, abs=5e-2)  # Permitimos un pequeño margen de error