import os
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

BASE_DIR = r"MASATI-v2"
# Carpeta donde se guardarán las imágenes generadas
OUT_DIR = os.path.join(BASE_DIR, "resultados_IoU")
os.makedirs(OUT_DIR, exist_ok=True)

def calcular_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def leer_cajas_xml(xml_path):
    cajas = []
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
                cajas.append([xmin, ymin, xmax, ymax])
    return cajas

def comparar_y_guardar(num_muestras=5):
    carpetas_barcos = ['ship', 'multi', 'coast_ship']
    imagenes_procesadas = []

    # 1. Encontrar imágenes que tengan tanto el XML original como el de SAM
    for carpeta in carpetas_barcos:
        img_dir = os.path.join(BASE_DIR, carpeta)
        xml_sam_dir = os.path.join(BASE_DIR, f"{carpeta}_labels_SAM")
        
        if not os.path.exists(xml_sam_dir):
            continue
            
        for xml_name in os.listdir(xml_sam_dir):
            if xml_name.endswith('.xml'):
                img_name = os.path.splitext(xml_name)[0] + ".png" 
                img_path = os.path.join(img_dir, img_name)
                
                if not os.path.exists(img_path):
                    img_name = os.path.splitext(xml_name)[0] + ".jpg"
                    img_path = os.path.join(img_dir, img_name)
                
                if os.path.exists(img_path):
                    imagenes_procesadas.append((carpeta, xml_name, img_path))

    if not imagenes_procesadas:
        print("¡ATENCIÓN! No se encontraron archivos XML en las carpetas '_SAM'.")
        return

    print(f"Se encontraron {len(imagenes_procesadas)} imágenes con etiquetas de SAM.")
    muestras = random.sample(imagenes_procesadas, min(num_muestras, len(imagenes_procesadas)))
    
    print("-" * 50)
    print(f"{'IMAGEN':<20} | {'CATEGORÍA':<10} | {'VALOR IoU'}")
    print("-" * 50)
    
    # 2. Calcular IoU, GUARDAR y MOSTRAR las imágenes
    for carpeta, xml_name, img_path in muestras:
        xml_orig_path = os.path.join(BASE_DIR, f"{carpeta}_labels", xml_name)
        xml_sam_path = os.path.join(BASE_DIR, f"{carpeta}_labels_SAM", xml_name)
        
        cajas_orig = leer_cajas_xml(xml_orig_path)
        cajas_sam = leer_cajas_xml(xml_sam_path)
        
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img_rgb)
        
        texto_iou = []
        
        for i in range(min(len(cajas_orig), len(cajas_sam))):
            box_orig = cajas_orig[i]
            box_sam = cajas_sam[i]
            
            iou = calcular_iou(box_orig, box_sam)
            texto_iou.append(f"{iou:.2f}")
            
            # Caja original (ROJO - línea punteada)
            w_orig = box_orig[2] - box_orig[0]
            h_orig = box_orig[3] - box_orig[1]
            rect_orig = patches.Rectangle((box_orig[0], box_orig[1]), w_orig, h_orig, 
                                          linewidth=2, edgecolor='red', facecolor='none', linestyle='dashed', label='Original (Humano)' if i==0 else "")
            ax.add_patch(rect_orig)
            
            # Caja SAM (VERDE - línea sólida)
            w_sam = box_sam[2] - box_sam[0]
            h_sam = box_sam[3] - box_sam[1]
            rect_sam = patches.Rectangle((box_sam[0], box_sam[1]), w_sam, h_sam, 
                                         linewidth=2, edgecolor='green', facecolor='none', label='SAM (IA)' if i==0 else "")
            ax.add_patch(rect_sam)

        str_ious = ", ".join(texto_iou)
        plt.title(f"Imagen: {os.path.basename(img_path)} | IoU: {str_ious}")
        plt.legend()
        plt.axis('off')
        
        nombre_salida = f"comparacion_{os.path.basename(img_path)}"
        ruta_salida = os.path.join(OUT_DIR, nombre_salida)
        
        # 1. Guarda la imagen en la carpeta
        plt.savefig(ruta_salida, bbox_inches='tight')
        
        # 2. Muestra la imagen en pantalla
        plt.show()
        
        # 3. Cierra la figura para evitar sobrecarga de memoria
        plt.close(fig) 
        
        # Imprimir fila para tu tabla
        print(f"{os.path.basename(img_path):<20} | {carpeta:<10} | {str_ious}")

    print("-" * 50)
    print(f"¡Listo! Las imágenes comparativas se guardaron en la carpeta: {OUT_DIR}")

comparar_y_guardar(num_muestras=5)