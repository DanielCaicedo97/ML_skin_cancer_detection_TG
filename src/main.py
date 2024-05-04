import cv2
import os
from segmentation.main import Segmentation
from pre_procesing.main import PreProcessing
from extract_features.texture_features import TextureFeatures
from extract_features.geometry_features import GeometryFeatures
from extract_features.color_features import ColorFeatures

if __name__ == "__main__":

    path_image = r'database\data_base_pre_processing\images\ISIC_0000128.jpg'
    path_mask = r'database\data_base_pre_processing\masks\ISIC_0000128_segmentation.png'
    img = cv2.imread(path_image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(path_mask,0)

    # Aplicar la máscara a la imagen
    segmented_image = cv2.bitwise_and(img, img, mask=mask)

    geo_feature = GeometryFeatures(mask)

    print(geo_feature._radial_variance())

    # #preprocessin all images get a new_data_preprocesing

    # path_all_data = r'C:\Users\USUARIO\Desktop\OTROS\DANIEL\TRABAJO GRADO\BASE_DE_DATOS\analisis_de_datos\database_final'
    # path_save_data = r'./src/masks/'
    # list_of_path_image = os.listdir(os.path.join(path_all_data,'masks'))

    # for image_path in list_of_path_image:
    #     try:
    #         image_path_to_Read = os.path.join(path_all_data,'masks', image_path)
    #         processed_img = PreProcessing(image_path_to_Read).pipeline_masks()
    #         if not os.path.isdir(path_save_data):
    #             os.mkdir(path_save_data)

    #         cv2.imwrite(os.path.join(path_save_data , image_path),processed_img)
    #         print('image {0} saved sucessfully'.format(image_path))
    #     except Exception as e: 
    #         print(e) 
            
    # #PreProcesssingq
    # img_path = 'src/data_example/raw/ISIC_1435135.JPG'  # Ruta de tu imagen
    # img = cv2.imread(img_path,cv2.COLOR_BGR2RGB)
    # preprocessor = PreProcessing(img_path)
    # processed_img = preprocessor.pipeline_preprocessing()

    # img_pre = preprocessor._resize(img, max_dimension=512)
    # mask = preprocessor._apply_black_hat(img_pre)
    # img_pre = preprocessor._apply_inpainting(img_pre,mask)
    # img_pre = preprocessor._gauss_filter(img_pre)
    # img_pre = preprocessor._sharpen_filter(img_pre)

    # # Mostrar la imagen procesada
    # cv2.imshow('Imagen sin Procesar', img)
    # cv2.imshow('black-hat',mask)
    # cv2.imshow('Imagen Procesada', img_pre)


    # print(img.shape)

    # print(img_pre.shape)
    # #Segmentation
    
    # segmentator = Segmentation(processed_img)

    # # Segmentación usando método de Otsu
    # otsu_segmented_img = segmentator.method_otsu()

    # # Segmentación usando K-Means
    # kmeans_segmented_img = segmentator.k_means_segmentation()

    # # Mostrar las imágenes segmentadas
    # cv2.imshow('Otsu Segmentation', otsu_segmented_img)
    # cv2.imshow('K-Means Segmentation', kmeans_segmented_img)

    # texture_features = TextureFeatures(kmeans_segmented_img)
    # geometry_features = GeometryFeatures(otsu_segmented_img)
    # color_features = ColorFeatures(processed_img)

    # print(texture_features.get_texture_features())
    # print(geometry_features.get_geometry_features())
    # print(color_features.get_color_features())

    # geometry_features.fractal_dimension_graphic()

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()