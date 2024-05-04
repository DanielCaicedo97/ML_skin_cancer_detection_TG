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
        radial_variance = self._radial_variance()
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
            'radial_variance': radial_variance,
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

    def _transformation_to_center(self, binary_mask):
               # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
        
        return shifted , center, axes , angle

    def _extract_contour_max(self, binary_mask):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)
        # Crear una imagen vacía del mismo tamaño que la imagen original
        contour_image = np.zeros_like(binary_mask)
        # Dibujar el contorno en la imagen vacía
        cv2.drawContours(contour_image, [contour], -1, (255, 255, 255), thickness=1)

        return contour_image

    def _radial_variance_calc_by_contour(self, contour, center):
 

        # Lista para almacenar los puntos del contorno
        list_points = []

        # Obtener las dimensiones del contorno
        H, L = contour.shape

        # Recorrer el contorno para encontrar los puntos
        for i in range(H):
            for j in range(L):
                # Verificar si el valor del punto es diferente de cero (es parte del contorno)
                if contour[i][j] != 0:
                    # Agregar las coordenadas del punto a la lista
                    list_points.append([i, j])

        # Calcular la distancia desde el centro a cada punto
        distances = [np.linalg.norm(np.array(point) - np.array(center)) for point in list_points]

        # Calcular la media de las distancias euclidianas
        mean_distance = np.mean(distances)
        # Calcular el perímetro del contorno
        perimeter = len(list_points)
        # Calcular la varianza radial
        radial_variance = np.sum([(distance - mean_distance) ** 2 for distance in distances]) / (mean_distance ** 2 * perimeter)

        return radial_variance 
    
    def _radial_variance(self):

        #calcular varianza radial de la mascara 
        mask, center, axes , angle  = self._transformation_to_center(self.binary_mask)
        contour = self._extract_contour_max(mask)
        # Convertir el centro a valores enteros
        x, y = center
        x = int(x)
        y = int(y)
        center = (x, y)
        radial_variance_mask = self._radial_variance_calc_by_contour(contour, center)

        #calcular varianza radial de la circunferencia 
        circunferencia = np.zeros_like(mask)
        cv2.circle(circunferencia,np.array(center),int(axes[1]/2),(255, 255, 255),1)
        radial_variance_circle = self._radial_variance_calc_by_contour(circunferencia, center)

        radial_variance_total = radial_variance_mask/radial_variance_circle
        return  radial_variance_total     

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


    def draw_ellipse(self, img):
         # Calcular el contorno de la máscara binaria
        contours, _ = cv2.findContours(self.binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Obtener el contorno más grande
        contour = max(contours, key=cv2.contourArea)

        # Ajustar una elipse al contorno
        ellipse = cv2.fitEllipse(contour)

        # Extraer los parámetros de la elipse
        center, axes, angle = ellipse
        # Dibujar el centro de la elipse
        center = (int(center[0]), int(center[1]))  # Convertir el centro a coordenadas enteras
        cv2.circle(img, center, 5, (0, 0, 255), -1)  # Dibujar un círculo rojo en el centro
        # Dibujar los ejes de la elipse
        # Calcular el punto en el borde de la elipse para el eje mayor
        # Calcular el punto en el borde de la elipse para el eje mayor
        endpoint_x = int(center[0] + axes[1] / 2 * np.cos(np.radians(270+ angle)))
        endpoint_y = int(center[1] + axes[1] / 2 * np.sin(np.radians(270+angle)))

        # Calcular el punto en el borde opuesto de la elipse para el eje mayor
        opposite_endpoint_x = int(center[0] - axes[1] / 2 * np.cos(np.radians(270+angle)))
        opposite_endpoint_y = int(center[1] - axes[1] / 2 * np.sin(np.radians(270+angle)))

        # Dibujar el eje mayor de la elipse
        cv2.line(img, (endpoint_x, endpoint_y), (opposite_endpoint_x, opposite_endpoint_y), (255, 0, 0), 2)

        cv2.ellipse(img, ellipse, (0, 255, 0), 2)  # Dibujar la elipse en verde con un grosor de 2

        # Mostrar la imagen con el centro y los ejes dibujados
        cv2.imshow('Imagen con elipse', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_radial_variance(self):
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

        # Llevar a cabo la traslación
        translation_matrix = np.float32([[1, 0, center_diff[0]], [0, 1, center_diff[1]]])
        shifted = cv2.warpAffine(rotated_img, translation_matrix, (rotated_img.shape[1], rotated_img.shape[0]))
        shifted = cv2.cvtColor(shifted,cv2.COLOR_GRAY2RGB)

        cv2.circle(shifted,image_center,int(axes[1]/2),(0, 255, 0),2)
        cv2.circle(shifted,image_center,2,(0, 255, 0),2)
        # Mostrar las imágenes
        self._show_images(self.binary_mask,shifted)
        

    def draw_rotation_and_shifted(self):
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

        # Llevar a cabo la traslación
        translation_matrix = np.float32([[1, 0, center_diff[0]], [0, 1, center_diff[1]]])
        shifted = cv2.warpAffine(rotated_img, translation_matrix, (rotated_img.shape[1], rotated_img.shape[0]))

        # Realizar flip horizontal y vertical respecto al centro de la imagen
        flip_horizontal = cv2.flip(shifted, 1)
        flip_vertical = cv2.flip(shifted, 0)

        # Encontrar los contornos en las imágenes original y reflejadas
        contours_center_image, _ = cv2.findContours(shifted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_flip_horizontal, _ = cv2.findContours(flip_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_flip_vertical, _ = cv2.findContours(flip_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crear imágenes vacías del mismo tamaño que la imagen original
                
        contour_image_shifted = np.zeros_like(shifted)
        contour_image_shifted = cv2.cvtColor(contour_image_shifted, cv2.COLOR_GRAY2RGB)

        contour_image_horizontal = np.zeros_like(shifted)
        contour_image_horizontal = cv2.cvtColor(contour_image_horizontal, cv2.COLOR_GRAY2RGB)

        contour_image_vertical = np.zeros_like(shifted)
        contour_image_vertical = cv2.cvtColor(contour_image_vertical, cv2.COLOR_GRAY2RGB)

        rotated_img = cv2.cvtColor(rotated_img,cv2.COLOR_GRAY2RGB)
        shifted = cv2.cvtColor(shifted,cv2.COLOR_GRAY2RGB)
        shifted_without_lines = shifted.copy()
               # Realizar flip horizontal y vertical respecto al centro de la imagen
        flip_horizontal =  cv2.cvtColor(flip_horizontal,cv2.COLOR_GRAY2RGB)
        flip_vertical =  cv2.cvtColor(flip_vertical,cv2.COLOR_GRAY2RGB)

        # Dibujar los contornos en las imágenes vacías
        cv2.drawContours(contour_image_shifted, contours_center_image, -1, (255, 255, 255), thickness=1)

        cv2.drawContours(contour_image_horizontal, contours_center_image, -1, (255, 255, 255), thickness=1)
        cv2.drawContours(contour_image_horizontal, contours_flip_horizontal, -1, (255, 255, 255), thickness=1)

        cv2.drawContours(contour_image_vertical, contours_center_image, -1, (255, 255, 255), thickness=1)
        cv2.drawContours(contour_image_vertical, contours_flip_vertical, -1, (255, 255, 255), thickness=1)

        # Dibujar líneas en las imágenes rotadas y trasladadas
        self._draw_lines(rotated_img)
        self._draw_lines(shifted)
        self._draw_lines(flip_horizontal)
        self._draw_lines(flip_vertical)

        # Mostrar las imágenes
        self._show_images(self.binary_mask, rotated_img, shifted, flip_horizontal, flip_vertical, contour_image_horizontal, contour_image_vertical,contour_image_shifted,shifted_without_lines)

    def _draw_lines(self, img):
        # Calcular el centro de la imagen
        height, width = img.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # Dibujar líneas horizontales y verticales
        cv2.line(img, (0, center_y), (width-1, center_y), (0, 255, 0), thickness=2)
        cv2.line(img, (center_x, 0), (center_x, height-1), (0, 255, 0), thickness=2)

    def _show_images(self, *images):
        for i, img in enumerate(images):
            cv2.imshow(f'Image {i}', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    def draw_boxes_fractal_dimension(self, boxes=2):
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

        # Llevar a cabo la traslación
        translation_matrix = np.float32([[1, 0, center_diff[0]], [0, 1, center_diff[1]]])
        shifted = cv2.warpAffine(rotated_img, translation_matrix, (rotated_img.shape[1], rotated_img.shape[0]))
        # Encontrar los contornos en las imágenes original y reflejadas
        contours_center_image, _ = cv2.findContours(shifted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Crear imágenes vacías del mismo tamaño que la imagen original

        contour_image_shifted = np.zeros_like(shifted)
        contour_image_shifted = cv2.cvtColor(contour_image_shifted, cv2.COLOR_GRAY2RGB)

        # Dibujar los contornos en las imágenes vacías
        cv2.drawContours(contour_image_shifted, contours_center_image, -1, (255, 255, 255), thickness=1)

        # Calcular las dimensiones del grid
        height, width = contour_image_shifted.shape[:2]
        box_width = width // boxes
        box_height = height // boxes

        # Dibujar líneas horizontales
        for i in range(1, boxes):
            y = i * box_height
            cv2.line(contour_image_shifted, (0, y), (width, y), (0, 255, 0), thickness=1)

        # Dibujar líneas verticales
        for i in range(1, boxes):
            x = i * box_width
            cv2.line(contour_image_shifted, (x, 0), (x, height), (0, 255, 0), thickness=1)

        self._show_images(contour_image_shifted)


