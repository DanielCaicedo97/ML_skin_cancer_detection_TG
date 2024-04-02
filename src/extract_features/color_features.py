import cv2

class ColorFeatures():
    def __init__(self, img):
        self.img = img

    def get_color_features(self):
        # Obtener características de color
        mean = self._mean_rgb()
        variance = self._variance_rgb()
        std_deviation = self._std_deviation_rgb()

        # Agrupar todas las características en un solo diccionario
        color_features = {**mean, **variance, **std_deviation}

        return color_features

    def _mean_rgb(self):
        # Calcular la media de cada canal de color RGB
        mean_b = cv2.mean(self.img[:,:,0])[0]
        mean_g = cv2.mean(self.img[:,:,1])[0]
        mean_r = cv2.mean(self.img[:,:,2])[0]

        # Crear diccionario con las medias de cada canal de color RGB
        mean = {
            'mean_r': mean_r,
            'mean_g': mean_g,
            'mean_b': mean_b
        }

        return mean

    def _variance_rgb(self):
        # Calcular la varianza de cada canal de color RGB
        variance_b = cv2.meanStdDev(self.img[:,:,0])[1][0]**2
        variance_g = cv2.meanStdDev(self.img[:,:,1])[1][0]**2
        variance_r = cv2.meanStdDev(self.img[:,:,2])[1][0]**2

        # Crear diccionario con las varianzas de cada canal de color RGB
        variance = {
            'variance_r': variance_r[0],
            'variance_g': variance_g[0],
            'variance_b': variance_b[0]
        }

        return variance

    def _std_deviation_rgb(self):
        # Calcular la desviación estándar de cada canal de color RGB
        std_dev_b = cv2.meanStdDev(self.img[:,:,0])[1][0]
        std_dev_g = cv2.meanStdDev(self.img[:,:,1])[1][0]
        std_dev_r = cv2.meanStdDev(self.img[:,:,2])[1][0]

        # Crear diccionario con las desviaciones estándar de cada canal de color RGB
        std_deviation = {
            'std_deviation_r': std_dev_r[0],
            'std_deviation_g': std_dev_g[0],
            'std_deviation_b': std_dev_b[0]
        }

        return std_deviation
