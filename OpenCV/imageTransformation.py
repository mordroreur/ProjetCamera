import cv2
import os

def contour_detection_and_save(input_folder, output_folder):
    # Vérifier si le dossier de sortie existe, sinon le créer
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Liste des fichiers dans le dossier d'entrée
    files = os.listdir(input_folder)
    
    for file_name in files:
        # Ignorer les fichiers qui ne sont pas des images
        if not (file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png')):
            continue
        
        # Chemin d'accès complet de l'image d'entrée
        input_path = os.path.join(input_folder, file_name)
        
        # Charger l'image
        image = cv2.imread(input_path)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Appliquer un seuillage adaptatif
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 5)

        # Inverser si nécessaire
        if True:
            thresh = cv2.bitwise_not(thresh)
        
        # Enregistrer l'image transformée dans le dossier de sortie
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, thresh)

# Appel de la fonction avec le dossier d'entrée et de sortie spécifiés
contour_detection_and_save("../test-export/training", "../test_import/training")