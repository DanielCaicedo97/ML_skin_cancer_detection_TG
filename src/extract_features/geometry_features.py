import cv2 
import numpy  as np
import porespy as ps
import statsmodels.api as sm
import matplotlib.pyplot as plt

class GeometryFeatures():
    def __init__(self, binary_mask):
        self.binary_mask = binary_mask

    # geometry features asymetry, elipse and border 

    def get_geometry_features(self) -> dict:
        symmetry = self._symmetry()
        compactness_index = self._compactness_index()
        fractal_dimension = self._fractal_dimension()
        hu_moments = self._hu_moments()
                # Organizar los momentos de Hu en un diccionario
        hu_dict = {}
        for i in range(7):
            hu_dict[f'hu_{i+1}'] = hu_moments[i][0]
         # Combinar todos los diccionarios en uno solo
        geometry_features_dict = {
            'symmetry_Horizontal': symmetry[0],
            'symmetry_vertical': symmetry[1],
            'compactness_index': compactness_index,
            'fractal_dimension': fractal_dimension,
            **hu_dict
        }
        return geometry_features_dict

    def _symmetry(self):
        # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Ajustar una elipse al contorno
        ellipse = cv2.fitEllipse(contour)
        # Obtener los parámetros de la elipse
        center, axes, angle = ellipse

        # Rotar la imagen original para que el eje principal se posicione a 90°
        rotation_matrix = cv2.getRotationMatrix2D(center, angle - 180, 1.0)
        rotated_img = cv2.warpAffine(self.binary_mask, rotation_matrix, self.binary_mask.shape[1::-1], flags=cv2.INTER_LINEAR)

        # Calcular la diferencia entre el centro de la imagen y el centro de la elipse
        image_center = (self.binary_mask.shape[1] // 2, self.binary_mask.shape[0] // 2)
        center_diff = (image_center[0] - center[0], image_center[1] - center[1])

        translation_matrix =  np.float32([[1, 0, center_diff[0]], [0, 1, center_diff[1]]])
         # Llevamos a cabo la transformación.
        shifted = cv2.warpAffine(rotated_img, translation_matrix, (rotated_img.shape[1], rotated_img.shape[0]))
        # Realizar flip horizontal y vertical respecto al centro de la imagen
        flip_horizontal = cv2.flip(shifted, 1)
        flip_vertical = cv2.flip(shifted, 0)

        # Calcular la intersección entre la imagen original y el flip horizontal y vertical
        intersection_horizontal = cv2.bitwise_and(shifted, flip_horizontal)
        intersection_vertical = cv2.bitwise_and(shifted, flip_vertical)

        # Calcular el área de la intersección y el área total de la imagen
        area_intersection_horizontal = np.sum(intersection_horizontal)
        area_intersection_vertical = np.sum(intersection_vertical)
        total_area = np.sum(shifted)

        # Calcular la simetría horizontal y vertical
        horizontal_symmetry = area_intersection_horizontal / total_area
        vertical_symmetry = area_intersection_vertical / total_area
        
        return np.array([horizontal_symmetry, vertical_symmetry])

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
        compactness_index = (perimeter ** 2) / (4 *np.pi* area)

        return compactness_index

    def _fractal_dimension(self):
        # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)
        # Crear una imagen vacía del mismo tamaño que la imagen original
        contour_image = np.zeros_like(self.binary_mask)
        # Dibujar el contorno en la imagen vacía
        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=1)
        # Calcular la dimensión fractal utilizando poreSpy
        # Generar una serie de datos de 2 a 20
        bins = np.arange(2, 51)    
        data = ps.metrics.boxcount(contour_image,bins=bins)
        # ajuste de datos en escala logaritmica
        x = np.log(data.size)
        y = np.log(data.count)
        # Agregar una constante a x para ajustar una línea recta con intercepto en el origen
        x_with_const = sm.add_constant(x)

        # Ajustar el modelo de regresión lineal
        model = sm.OLS(y, x_with_const)
        # Ajustar el modelo a los datos
        results = model.fit()
        # Obtener los coeficientes (pendiente e intercepto)
        slope = results.params[1]

        return np.abs(slope)
    
    def fractal_dimension_graphic(self):
         # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)
        # Crear una imagen vacía del mismo tamaño que la imagen original
        contour_image = np.zeros_like(self.binary_mask)
        # Dibujar el contorno en la imagen vacía
        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=1)
        # Calcular la dimensión fractal utilizando poreSpy
        # Generar una serie de datos de 2 a 20
        bins = np.arange(2, 51)    
        data = ps.metrics.boxcount(contour_image,bins=bins)
        # ajuste de datos en escala logaritmica
        x = np.log(data.size)
        y = np.log(data.count)
        # Graficar los resultados
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('Tamaño del borde de la caja')
        ax1.set_ylabel('Número de cajas que abarcan fases')
        ax2.set_xlabel('Tamaño del borde de la caja')
        ax2.set_ylabel('Pendiente')
        ax2.set_xscale('log')
        ax1.plot(data.size, data.count, '-o')
        ax2.plot(x, y, '-o')
        plt.show()

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

        return hu_moments