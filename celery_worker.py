import celery
import os
import shutil

from celery import Celery
from distutils.dir_util import copy_tree

from config import broker_url, data_folder, settings
from inference import predict_safe_regions

app = Celery('Crop_Field_Segmentation', broker=broker_url, backend='rpc://')
app.config_from_object(settings.CELERY.config)

folder_path = '/usr/src/app/temp'
resultant_file_paths = os.listdir(folder_path)
# resultant_file_paths = []

# for file in safe_file_paths:
#     keywords = [f'20211{x}' for x in [
#         '024',
#         '025',
#         '026',
#         '027',
#         '028',
#         '029',
#         '030',
#         '031',
#         '101'
#     ]]
#
#     for keyword in keywords:
#         if keyword in file:
#             cur_file_path = os.path.join(folder_path, file)
#             resultant_file_paths.append(cur_file_path)
            
count = 0
total = len(resultant_file_paths)

for file_path in resultant_file_paths:
    file_name = file_path.split('/')[-1]
    files = os.listdir(os.path.join(folder_path, file_path))

    num_files = len(os.listdir(data_folder['04_test'][file_name].path))

    if num_files < 4:
        for file in files:
            if os.path.exists(data_folder['04_test'][file_name][file]):
                continue
                
            shutil.copyfile(os.path.join(folder_path, file_path, file), data_folder['04_test'][file_name][file])
            
    count += 1

    print(f"Copied {count} folders out of {total}.")

resultant_file_paths = os.listdir(data_folder['04_test'])

for file_path in resultant_file_paths:
    file_name = file_path.split('/')[-1]
    files = os.listdir(os.path.join(folder_path, file_path))
    
    # print(f"{file_name}: {os.listdir(os.path.join(folder_path, file_path))}")
    
    if count < 4:
        print(f"{file_name}: {os.listdir(os.path.join(folder_path, file_path))}")

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
