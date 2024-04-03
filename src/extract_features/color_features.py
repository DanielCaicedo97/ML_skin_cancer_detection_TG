import cv2

class ColorFeatures():
    def __init__(self, img):
        self.img = img
    def get_color_features(self):
        # Obtener características de color
        mean_rgb = self._mean_rgb()
        variance_rgb = self._variance_rgb()
        std_deviation_rgb = self._std_deviation_rgb()

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


    def _mean_rgb(self):
        # Organizar los canales RGB
        b, g, r = cv2.split(self.img)

        # Calcular la media de cada canal de color RGB
        mean_r = cv2.mean(r)[0]
        mean_g = cv2.mean(g)[0]
        mean_b = cv2.mean(b)[0]

        # Retornar los valores de RGB
        return mean_r, mean_g, mean_b

    def _variance_rgb(self):
        # Organizar los canales RGB
        b, g, r = cv2.split(self.img)

        # Calcular la varianza de cada canal de color RGB
        variance_r = cv2.meanStdDev(r)[1][0]**2
        variance_g = cv2.meanStdDev(g)[1][0]**2
        variance_b = cv2.meanStdDev(b)[1][0]**2

        # Retornar los valores de RGB
        return variance_r, variance_g, variance_b

    def _std_deviation_rgb(self):
        # Organizar los canales RGB
        b, g, r = cv2.split(self.img)

        # Calcular la desviación estándar de cada canal de color RGB
        std_dev_r = cv2.meanStdDev(r)[1][0]
        std_dev_g = cv2.meanStdDev(g)[1][0]
        std_dev_b = cv2.meanStdDev(b)[1][0]

        # Retornar los valores de RGB
        return std_dev_r, std_dev_g, std_dev_b
