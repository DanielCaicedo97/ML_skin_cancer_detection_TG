import PIL.ImageOps
import cv2
import numpy as np
import PIL 

class PreProcessing():
    def __init__(self, img_path):
        self.img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)

    def pipeline_preprocessing(self):
        self.img = self._resize(self.img, 512)
        self.img = self._apply_hair_removing(self.img)
        self.img = self._gauss_filter(self.img)
        self.img = self._sharpen_filter(self.img)
        return self.img

    def _gauss_filter(self, img):
        # Aplicar un filtro gaussiano
        img_gaussian = cv2.GaussianBlur(img, (9,9), 0)
        return img_gaussian

    def _sharpen_filter(self, img , alfa=0.5):
        # Aplicar un filtro de enfoque (sharpening)

        kernel_sharpening = np.array([[-alfa,alfa-1,-alfa], [alfa-1, alfa + 5,alfa-1],[-alfa,alfa-1,-alfa]])*(1/(alfa+1))
        img_sharpened = cv2.filter2D(img, -1, kernel_sharpening)
        return img_sharpened

    def _resize(self, img, max_dimension):
        """
        Redimensiona una imagen manteniendo la relación de aspecto.

        Args:
            img (ndarray): Imagen de entrada.
            max_dimension (int): Dimensión máxima permitida.

        Returns:
            ndarray: Imagen redimensionada.
        """
    # Convertir la imagen de OpenCV a formato Pillow (RGB)
        img_pillow = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Redimensionar la imagen manteniendo la relación de aspecto usando Pillow
        img_resized_pillow = PIL.ImageOps.fit(img_pillow,(max_dimension, max_dimension))

        # Convertir la imagen de Pillow a formato OpenCV (BGR)
        img_resized = np.array(img_resized_pillow, dtype=np.uint8)
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

        return img_resized


    
    def _apply_hair_removing(self, img_rgb):
        img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_RGB2GRAY)
        # Kernel for the morphological filtering
        kernel = cv2.getStructuringElement(1,(17,17))
    
        # Perform the blackHat filtering on the grayscale image to find the hair countours
        blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)
        _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
        final_image = cv2.inpaint(img_rgb,threshold,1,cv2.INPAINT_TELEA)
        return final_image

# # Ejemplo de uso
# img_path = 'tu_imagen.jpg'  # Ruta de tu imagen
# preprocessor = PreProcessing(img_path)
# processed_img = preprocessor.pipeline_preprocessing()

# # Mostrar la imagen procesada
# cv2.imshow('Imagen Procesada', processed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
