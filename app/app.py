import diskcache
from functools import partial

from lgblkb_tools import Folder, logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from box import Box
import itertools as it
import more_itertools as mit
from lgblkb_tools.telegram_notify import TheChat
from telegram import Bot
from config import settings

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 25)

notify = partial(TheChat(r'-432356237', bot=Bot('1084990340:AAFCm3odQOHCPDTqTStT_KQwidCaOrXwJNc')).send_message)
project_folders = Box({k: Folder(v, assert_exists=True) for k, v in settings.project.dirs.items()}, frozen_box=True)
s2_storage_folder = project_folders.s2_storage_folder
cache_folder = project_folders.cache_folder
data_folder = project_folders.data_folder

cache = diskcache.Cache(cache_folder['diskcache'])


@logger.trace()
def main():
    pass


if __name__ == '__main__':
    main()
