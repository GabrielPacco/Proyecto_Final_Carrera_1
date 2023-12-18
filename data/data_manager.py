import os
import cv2
import logging

class DataManager:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_images_from_folder(self, folder_name, image_formats=['.jpg', '.jpeg', '.png']):
        """
        Carga todas las imágenes de una subcarpeta específica que coinciden con los formatos especificados.
        """
        folder_path = os.path.join(self.base_path, folder_name)
        images = []
        file_names = []
        for filename in os.listdir(folder_path):
            if any(filename.lower().endswith(ext) for ext in image_formats):
                img_path = os.path.join(folder_path, filename)
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        images.append(img)
                        file_names.append(filename)
                except Exception as e:
                    logging.error(f"Error al cargar la imagen {filename}: {e}")
        return images, file_names

    def load_all_images(self, image_formats=['.jpg', '.jpeg', '.png']):
        """
        Carga todas las imágenes de todas las subcarpetas que coinciden con los formatos especificados.
        """
        categories = os.listdir(self.base_path)
        all_images = {}
        for category in categories:
            category_path = os.path.join(self.base_path, category)
            if os.path.isdir(category_path):
                images, file_names = self.load_images_from_folder(category, image_formats)
                all_images[category] = (images, file_names)
        return all_images

    def save_preprocessed_image(self, image, file_name, output_folder):
        """
        Guarda la imagen preprocesada en una carpeta de salida específica.
        """
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            cv2.imwrite(os.path.join(output_folder, file_name), image)
        except Exception as e:
            logging.error(f"Error al guardar la imagen {file_name}: {e}")
