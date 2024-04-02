import cv2
import numpy as np 
import mahotas 

class TextureFeatures():

    def __init__(self, img_gray) -> None:
        self.img_gray = img_gray

    def get_texture_features(self) -> dict:
        haralick_features = self._haralick_Textures()
        texture_features_dict  = {**haralick_features}
        return texture_features_dict

    def _haralick_Textures(self):
        # Convertir la imagen a escala de grises
        gray_img = self.img_gray

        # Calcular las texturas de Haralick
        textures = mahotas.features.haralick(gray_img.astype(np.uint8))

        # Crear un diccionario con las características de Haralick
        haralick_textures = {
            'Contrast': textures[..., 1].mean(),
            'Correlation': textures[..., 2].mean(),
            'Energy': textures[..., 4].mean(),
            'Homogeneity': textures[..., 8].mean(),
            'Sum of squares variance': textures[..., 9].mean(),
            'Variance': textures[..., 10].mean(),
            'Sum average': textures[..., 11].mean(),
            'Entropy': textures[..., 12].mean()
        }

        return haralick_textures