import cv2 
import numpy

from geometry_features import GeometryFeatures
from texture_features import TextureFeatures
class Features():

    def __init__(self, img ,binary_mask):
        self.img = img
        self.img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        self.binary_mask = binary_mask
    
    def geometry_features(self)->dict:
        geometry_features = GeometryFeatures(self.binary_mask).get_geometry_features()
        return geometry_features
    
    def texture_features(self):
        texture_features = TextureFeatures(self.img_gray).get_texture_features()
        return texture_features
        
    def color_features():
        pass
