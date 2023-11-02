# Detection-of-Melanoma
Detection of Melanoma(image preprocessing)
import cv2
import os
import numpy as np
import pathlib

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (180, 180))
    
    # contrast enhancement
    img_yuv = cv2.cvtColor(image_resized, cv2.COLOR_RGB2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
   
    # noise reduction
    image = cv2.GaussianBlur(image, (5, 5), 0)
   
    # Hair removal
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = np.ones((2,2), np.uint8)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    hairs = cv2.inRange(tophat, 10, 255)
    image = cv2.inpaint(image, hairs, inpaintRadius=1, flags=cv2.INPAINT_TELEA)
   
    return image

def preprocess_and_save_images(source_folder, destination_folder):
    main_subdirs = ["Train", "Test"]
    categories = ["melanoma", "non-melanoma"]
    
    print("Starting preprocessing...")
    
    for main_subdir in main_subdirs:
        for category in categories:
            source_category_path = source_folder.joinpath(main_subdir, category)
            destination_category_path = destination_folder.joinpath(main_subdir, category)
            
            print(f"Checking source category path: {source_category_path}")

            # Ensure the source category path exists
            if not source_category_path.exists():
                print(f"Warning: Source category path {source_category_path} doesn't exist!")
                continue
            
            # Ensure the destination category path exists
            if not destination_category_path.exists():
                print(f"Creating directory: {destination_category_path}")
                destination_category_path.mkdir(parents=True)

            for filename in os.listdir(source_category_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    print(f"Processing file: {filename}")
                    source_image_path = source_category_path.joinpath(filename)
                    preprocessed_image = preprocess_image(str(source_image_path))
                    
                    destination_image_path = destination_category_path.joinpath(filename)
                    cv2.imwrite(str(destination_image_path), cv2.cvtColor(preprocessed_image, cv2.COLOR_RGB2BGR))
                    print(f"Saved preprocessed image to: {destination_image_path}")

if __name__ == '__main__':
    source_folder = pathlib.Path(r"C:/Users/zhu/Skin cancer ISIC The International Skin Imaging Collaboration")
    destination_folder = pathlib.Path(r"C:/Users/zhu/preprocess_image")
    preprocess_and_save_images(source_folder, destination_folder)
