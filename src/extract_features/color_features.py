import numpy as np
import matplotlib.pyplot as plt 
import cv2

class ColorFeatures():
    def __init__(self, img):
        self.img = img

    def get_color_features(self, mask):
        # Obtener características de color
        mean_rgb = self._mean_rgb(mask)
        variance_rgb = self._variance_rgb(mask)
        std_deviation_rgb = self._std_deviation_rgb(mask)

        # Agrupar todas las características en un solo diccionario
        color_features = {
            'mean_r': mean_rgb[0],
            'mean_g': mean_rgb[1],
            'mean_b': mean_rgb[2],
            'variance_r': variance_rgb[0],
            'variance_g': variance_rgb[1],
            'variance_b': variance_rgb[2],
            'std_deviation_r': std_deviation_rgb[0],
            'std_deviation_g': std_deviation_rgb[1],
            'std_deviation_b': std_deviation_rgb[2]
        }

        return color_features

    def _mean_rgb(self, mask):
        # Aplicar la máscara a la imagen
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Organizar los canales RGB de la imagen mascarada
        b, g, r = cv2.split(masked_img)

        # Calcular la media de cada canal de color RGB
        mean_r = np.mean(r[r != 0]) if np.any(r != 0) else 0
        mean_g = np.mean(g[g != 0]) if np.any(g != 0) else 0
        mean_b = np.mean(b[b != 0]) if np.any(b != 0) else 0

        # Retornar los valores de la media para RGB
        return mean_r, mean_g, mean_b

    def _variance_rgb(self, mask):
        # Aplicar la máscara a la imagen
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Organizar los canales RGB de la imagen mascarada
        b, g, r = cv2.split(masked_img)

        # Calcular la varianza de cada canal de color RGB
        variance_r = np.var(r[r != 0]) if np.any(r != 0) else 0
        variance_g = np.var(g[g != 0]) if np.any(g != 0) else 0
        variance_b = np.var(b[b != 0]) if np.any(b != 0) else 0

        # Retornar los valores de la varianza para RGB
        return variance_r, variance_g, variance_b

    def _std_deviation_rgb(self, mask):
        # Aplicar la máscara a la imagen
        masked_img = cv2.bitwise_and(self.img, self.img, mask=mask)

        # Organizar los canales RGB de la imagen mascarada
        b, g, r = cv2.split(masked_img)

        # Calcular la desviación estándar de cada canal de color RGB
        std_dev_r = np.std(r[r != 0]) if np.any(r != 0) else 0
        std_dev_g = np.std(g[g != 0]) if np.any(g != 0) else 0
        std_dev_b = np.std(b[b != 0]) if np.any(b != 0) else 0

        # Retornar los valores de la desviación estándar para RGB
        return std_dev_r, std_dev_g, std_dev_b

    def draw_histogram(self, mask):
        # Definir los colores y crear una figura
        color = ('r', 'g', 'b')
        plt.figure()

        # Iterar sobre cada canal de color
        for i, col in enumerate(color):
            # Calcular el histograma para cada canal con la máscara dada
            histr = cv2.calcHist([self.img], [i], mask, [256], [0, 256])
            plt.plot(histr, color=col)

        # Configurar los límites de los ejes
        plt.xlim([0, 255])

        # Mostrar el histograma
        plt.show()

