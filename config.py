from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=['settings.yaml'],
    environments=True,
    load_dotenv=True,
)