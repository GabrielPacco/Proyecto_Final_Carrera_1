import os
import cv2

class DataManager:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_images_from_folder(self, folder_name):
        """
        Carga todas las imágenes en formato JPG de una subcarpeta específica.
        """
        folder_path = os.path.join(self.base_path, folder_name)
        images = []
        file_names = []
        for filename in os.listdir(folder_path):
            if filename.endswith('.JPG') or filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    file_names.append(filename)
        return images, file_names

    def load_all_images(self):
        """
        Carga todas las imágenes de todas las subcarpetas.
        """
        categories = os.listdir(self.base_path)
        all_images = {}
        for category in categories:
            category_path = os.path.join(self.base_path, category)
            if os.path.isdir(category_path):
                images, file_names = self.load_images_from_folder(category)
                all_images[category] = (images, file_names)
        return all_images

    def save_preprocessed_image(self, image, file_name, output_folder):
        """
        Guarda la imagen preprocesada en una carpeta de salida específica.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        cv2.imwrite(os.path.join(output_folder, file_name), image)
