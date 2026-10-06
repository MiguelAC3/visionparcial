import os
import cv2
import random
import numpy as np
import xml.etree.ElementTree as ET
import torch
from segment_anything import sam_model_registry, SamPredictor

# 1. Configurar modelo 
BASE_DIR = r"MASATI-v2"
sam_checkpoint = "C:/Users/migue/Desktop/wazaa/2026-2/vision parcial/sam_vit_b_01ec64.pth"

sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
sam.to(device="cpu")
predictor = SamPredictor(sam)

def procesar_y_guardar_nuevos_xml_muestra(limite=20):
    carpetas_barcos = ['ship', 'multi', 'coast_ship']
    todas_las_imagenes = []
    
    # 1. Recopilar la lista de todas las imágenes disponibles que tienen XML
    for carpeta in carpetas_barcos:
        img_dir = os.path.join(BASE_DIR, carpeta)
        xml_dir = os.path.join(BASE_DIR, f"{carpeta}_labels")
        
        # Asegurarnos de que las carpetas de destino existan
        nueva_carpeta_xml = os.path.join(BASE_DIR, f"{carpeta}_labels_SAM")
        os.makedirs(nueva_carpeta_xml, exist_ok=True)
        
        if not os.path.exists(img_dir) or not os.path.exists(xml_dir):
            continue
            
        for img_name in os.listdir(img_dir):
            if img_name.endswith(('.png', '.jpg')):
                xml_name = os.path.splitext(img_name)[0] + ".xml"
                if os.path.exists(os.path.join(xml_dir, xml_name)):
                    # Guardamos la tupla (carpeta, nombre_imagen)
                    todas_las_imagenes.append((carpeta, img_name))
    
    # 2. Seleccionar 20 imágenes al azar de toda la colección
    if len(todas_las_imagenes) > limite:
        imagenes_a_procesar = random.sample(todas_las_imagenes, limite)
    else:
        imagenes_a_procesar = todas_las_imagenes
        
    print(f"Se procesarán {len(imagenes_a_procesar)} imágenes aleatorias...")
    
    # 3. Procesar únicamente las imágenes seleccionadas
    for carpeta, img_name in imagenes_a_procesar:
        img_dir = os.path.join(BASE_DIR, carpeta)
        xml_dir = os.path.join(BASE_DIR, f"{carpeta}_labels")
        nueva_carpeta_xml = os.path.join(BASE_DIR, f"{carpeta}_labels_SAM")
        
        xml_name = os.path.splitext(img_name)[0] + ".xml"
        xml_path = os.path.join(xml_dir, xml_name)
        nuevo_xml_path = os.path.join(nueva_carpeta_xml, xml_name)
        
        print(f"Procesando: {img_name} (Categoría: {carpeta})")
        
        # Cargar imagen y preparar SAM
        img = cv2.imread(os.path.join(img_dir, img_name))
        if img is None: continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        
        # Leer XML original
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Procesar cada barco en la imagen
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            if bndbox is not None:
                xmin_orig = int(float(bndbox.find('xmin').text))
                ymin_orig = int(float(bndbox.find('ymin').text))
                xmax_orig = int(float(bndbox.find('xmax').text))
                ymax_orig = int(float(bndbox.find('ymax').text))
                
                input_box = np.array([xmin_orig, ymin_orig, xmax_orig, ymax_orig])
                
                masks, _, _ = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                mask = masks[0]
                # # --- NUEVAS LÍNEAS PARA MOSTRAR LA MÁSCARA BINARIA ---
                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(6,6))
                # # cmap='gray' fuerza a que los True se vean blancos y los False negros
                # plt.imshow(mask, cmap='gray') 
                # plt.title(f"Máscara Binaria SAM - {img_name}")
                # plt.axis('off')
                # plt.show()
                # # -----------------------------------------------------
                y_indices, x_indices = np.where(mask)
                
                if len(x_indices) > 0 and len(y_indices) > 0:
                    bndbox.find('xmin').text = str(np.min(x_indices))
                    bndbox.find('ymin').text = str(np.min(y_indices))
                    bndbox.find('xmax').text = str(np.max(x_indices))
                    bndbox.find('ymax').text = str(np.max(y_indices))
        
        # Guardar el nuevo XML
        tree.write(nuevo_xml_path)

    print("¡Proceso completado! Se han guardado las nuevas etiquetas ajustadas por SAM.")

# Ejecutar limitando a 20
procesar_y_guardar_nuevos_xml_muestra(limite=20)