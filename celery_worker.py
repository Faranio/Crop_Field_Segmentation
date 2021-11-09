import celery
import os
import shutil

from celery import Celery
from distutils.dir_util import copy_tree

from config import broker_url, data_folder, settings
# from inference import predict_safe_regions

# app = Celery('Crop_Field_Segmentation', broker=broker_url, backend='rpc://')
# app.config_from_object(settings.CELERY.config)

temp_folder = '/usr/src/app/temp'
target_folder = data_folder['04_test']


def copy_files_from_caches_to_data():
    temp_folder_files = os.listdir(temp_folder)
    
    for safe_folder in temp_folder_files:
        target_folder_name = target_folder[safe_folder]
    
        if os.path.exists(target_folder_name):
            num_files = len(os.listdir(target_folder_name))
    
            if num_files != 4:
                for file in os.listdir(target_folder_name):
                    os.remove(os.path.join(target_folder_name, file))
    
                for file in os.listdir(os.path.join(temp_folder, safe_folder)):
                    shutil.copyfile(os.path.join(temp_folder, safe_folder, file), os.path.join(target_folder_name, file))
        else:
            os.mkdir(os.path.join(target_folder_name))
    
            for file in os.listdir(os.path.join(temp_folder, safe_folder)):
                shutil.copyfile(os.path.join(temp_folder, safe_folder, file), os.path.join(target_folder_name, file))
    
    print("FINISHED COPYING!")
    
    
def check_data_files():
    files = os.listdir(target_folder)
    
    for file in files:
        file_path = os.path.join(target_folder, files)
        num_files = len(os.listdir(file_path))
        
        if num_files != 4:
            print(f"{file}: {os.listdir(file_path)}")
            
check_data_files()
    
    # print(f"{file_name}: {os.listdir(os.path.join(folder_path, file_path))}")

    # if os.path.exists(data_folder['03_results'][f'{file_name}.gpkg']):
    #     continue
    #
    # predict_safe_regions(
    #     safe_folder_path=file_path,
    #     tile_width=20000,
    #     tile_height=20000,
    #     confidence=0.8,
    #     intersection_threshold=0.5,
    #     save=True
    # )
