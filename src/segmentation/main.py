import cv2
import numpy as np

class Segmentation():
    def __init__(self, img):
        # Convertir la imagen a escala de grises
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def method_otsu(self):
        # Aplicar el método de Otsu para la segmentación
        _, segmented_img = cv2.threshold(self.img_gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invertir la máscara
        segmented_img = cv2.bitwise_not(segmented_img)
        return segmented_img

    def k_means_segmentation(self, k=2):
        # Preparar los datos para el algoritmo K-Means
        pixel_values = self.img_gray.reshape((-1, 1))
        pixel_values = np.float32(pixel_values)

        # Definir los criterios y aplicar el algoritmo K-Means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convertir los centros de cluster a valores enteros
        centers = np.uint8(centers)

        # Asignar a cada píxel el valor del centroide más cercano
        segmented_img = centers[labels.flatten()]
        segmented_img = segmented_img.reshape(self.img_gray.shape)

        return segmented_img


# # Ejemplo de uso
# img_path = 'tu_imagen.jpg'  # Ruta de tu imagen
# img = cv2.imread(img_path)

# segmentator = Segmentation(img)

# # Segmentación usando método de Otsu
# otsu_segmented_img = segmentator.method_otsu()

# # Segmentación usando K-Means
# kmeans_segmented_img = segmentator.k_means_segmentation()

# # Mostrar las imágenes segmentadas
# cv2.imshow('Otsu Segmentation', otsu_segmented_img)
# cv2.imshow('K-Means Segmentation', kmeans_segmented_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
