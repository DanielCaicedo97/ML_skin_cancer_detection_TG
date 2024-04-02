import cv2
import numpy as np

class PreProcessing():
    def __init__(self, img_path):
        self.img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)

    def pipeline_preprocessing(self):
        self.img = self._resize(self.img)
        self.img = self._gauss_filter(self.img)
        self.img = self._sharpen_filter(self.img)
        return self.img

    def _gauss_filter(self, img):
        # Aplicar un filtro gaussiano
        img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
        return img_gaussian

    def _sharpen_filter(self, img):
        # Aplicar un filtro de enfoque (sharpening)
        kernel_sharpening = np.array([[-1,0,-1], [0, 5,0],[-1,0,-1]])
        img_sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        return img_sharpened

    def _resize(self, img):
        # Obtener las dimensiones de la imagen original
        height, width = img.shape[:2]

        # Calcular el factor de escala para redimensionar manteniendo el aspecto original
        max_dimension = 512
        if width > height:
            scale_factor = max_dimension / width
        else:
            scale_factor = max_dimension / height

        # Redimensionar la imagen con el factor de escala calculado
        img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

        return img_resized



# # Ejemplo de uso
# img_path = 'tu_imagen.jpg'  # Ruta de tu imagen
# preprocessor = PreProcessing(img_path)
# processed_img = preprocessor.pipeline_preprocessing()

# # Mostrar la imagen procesada
# cv2.imshow('Imagen Procesada', processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
