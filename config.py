import box
import dynaconf
import lgblkb_tools

settings = dynaconf.Dynaconf(settings_files=['settings.yaml'], environments=True, load_dotenv=True)
project_folders = box.Box({k: lgblkb_tools.Folder(v, assert_exists=True) for k, v in settings.project.dirs.items()},
                          frozen_box=True)
cache_folder = project_folders.cache_folder
data_folder = project_folders.data_folder
test_folder = project_folders.test_folder
model_folder = data_folder['02_model']
model_path = model_folder['Max_Model.pt']
broker_url = settings.project.broker_url
