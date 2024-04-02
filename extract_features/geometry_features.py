import cv2 
import numpy  as np


class GeometryFeatures():
    def __init__(self, binary_mask):
        self.binary_mask = binary_mask

    # geometry features asymetry, elipse and border 

    def get_geometry_features(self) -> dict:
        asymmetry = self._asymmetry()
        compactness_index = self._compactness_index()
        fractal_dimension = self._fractal_dimension()
        hu_moments = self._hu_moments()
         # Combinar todos los diccionarios en uno solo
        geometry_features_dict = {**asymmetry, **compactness_index, **fractal_dimension, **hu_moments}
        return geometry_features_dict

    def _asymmetry(self):
        # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Ajustar una elipse al contorno
        ellipse = cv2.fitEllipse(contour)

        # Obtener los parámetros de la elipse
        center, axes, angle = ellipse

        # Rotar la imagen original para que el eje principal se posicione a 90°
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 90, 1.0)
        rotated_img = cv2.warpAffine(self.binary_mask, rotation_matrix, self.binary_mask.shape[1::-1], flags=cv2.INTER_LINEAR)
        
        # Dividir la imagen rotada en mitades horizontal y verticalmente
        height, width = rotated_img.shape[:2]
        half_height = height // 2
        half_width = width // 2
        top_half = rotated_img[:half_height, :]
        bottom_half = rotated_img[half_height:, :]
        left_half = rotated_img[:, :half_width]
        right_half = rotated_img[:, half_width:]

        # Calcular el nivel de simetría
        horizontal_symmetry = np.sum(np.abs(top_half - bottom_half)) / np.sum(rotated_img)
        vertical_symmetry = np.sum(np.abs(left_half - right_half)) / np.sum(rotated_img)

        asymmetry_dict = {
            'asymetry_Horizontal': horizontal_symmetry,
            'asymmetry_Vertical': vertical_symmetry
        }

        return asymmetry_dict

    def _compactness_index(self):
        # Encontrar contornos en la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si no se detectan contornos, devolver None
        if len(contours) == 0:
            return None

        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Calcular el perímetro y el área del contorno
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)

        # Calcular el índice de compacidad
        compactness_index = (perimeter ** 2) / (4 * np.pi * area)

        compactness_index_dict = {
            'compactness_index': compactness_index
        }

        return compactness_index_dict 


    def _fractal_dimension(self):
        pass


    def _hu_moments(self):
        # Encontrar contornos en la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Si no se detectan contornos, devolver None
        if len(contours) == 0:
            return None

        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Calcular momentos de Hu del contorno
        moments = cv2.moments(contour)
        hu_moments = cv2.HuMoments(moments)

        # Organizar los momentos de Hu en un diccionario
        hu_dict = {}
        for i in range(7):
            hu_dict[f'hu_{i+1}'] = hu_moments[i][0]

        return hu_dict