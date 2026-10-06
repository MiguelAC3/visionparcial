import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

# 1. Configura tu ruta base (la que ya te funcionó)
BASE_DIR = r"MASATI-v2"

def visualizar_imagenes_todas_categorias(num_imagenes=6):
    if not os.path.exists(BASE_DIR):
        print(f"Error: No se encuentra el directorio base '{BASE_DIR}'.")
        return

    # 2. Identificar cuáles son las carpetas de imágenes
    # Filtramos para obtener solo directorios y excluir los que terminan en '_labels'
    carpetas_imagenes = [
        d for d in os.listdir(BASE_DIR) 
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.endswith('_labels')
    ]
    
    todas_las_imagenes = []
    
    # 3. Recolectar todas las imágenes de todas las carpetas
    for carpeta in carpetas_imagenes:
        ruta_carpeta = os.path.join(BASE_DIR, carpeta)
        imagenes = [f for f in os.listdir(ruta_carpeta) if f.endswith('.png')]
        for img in imagenes:
            # Guardamos una tupla con (nombre_archivo, nombre_carpeta) para saber de dónde viene
            todas_las_imagenes.append((img, carpeta))
            
    if not todas_las_imagenes:
        print("No se encontraron imágenes en las subcarpetas.")
        return

    # 4. Seleccionar imágenes al azar de TODA la colección
    imagenes_seleccionadas = random.sample(todas_las_imagenes, min(num_imagenes, len(todas_las_imagenes)))
    
    for img_name, categoria in imagenes_seleccionadas:
        img_path = os.path.join(BASE_DIR, categoria, img_name)
        
        # Construir la posible ruta del archivo XML
        xml_dir = os.path.join(BASE_DIR, f"{categoria}_labels")
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(xml_dir, xml_name)
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer la imagen {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img_rgb)
        plt.title(f"Categoría: {categoria} | Archivo: {img_name}")
        
        # 5. Dibujar recuadros solo si existe el archivo XML correspondiente
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin = int(float(bndbox.find('xmin').text))
                    ymin = int(float(bndbox.find('ymin').text))
                    xmax = int(float(bndbox.find('xmax').text))
                    ymax = int(float(bndbox.find('ymax').text))
                    
                    width = xmax - xmin
                    height = ymax - ymin
                    
                    rect = patches.Rectangle((xmin, ymin), width, height, 
                                             linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
        else:
            print(f"Nota: Sin etiquetas XML para {img_name} (Posible clase sin barcos)")
        
        plt.axis('off')
        plt.show()

# Ejecutar la visualización (puedes cambiar la cantidad de imágenes a mostrar)
visualizar_imagenes_todas_categorias(num_imagenes=6)