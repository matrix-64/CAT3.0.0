from tqdm.auto import tqdm
from tqdm import trange
import time

with tqdm(trange(12), total=12, desc='Importing modules', ncols=70, ascii=' =', leave=True) as pbar:
    pbar.set_description('Importing modules...')
    pbar.refresh()
    pbar.update(5)
    pbar.set_description('Importing keras...')
    pbar.refresh()
    import keras
    pbar.update(1)
    pbar.set_description('Importing numpy...')
    pbar.refresh()
    import numpy as np
    pbar.update(1)
    pbar.set_description('Importing opencv...')
    pbar.refresh()
    import cv2
    pbar.update(1)
    pbar.set_description('Importing mediapipe...')
    pbar.refresh()
    import mediapipe as mp
    pbar.update(1)
    pbar.set_description('Importing pyautogui...')
    pbar.refresh()
    import pyautogui as pg
    pbar.update(1)
    pbar.set_description('Importing mmact...')
    pbar.refresh()
    from mmcat import mouseModeCAT
    pbar.update(1)
    pbar.set_description('Importing kmcat...')
    pbar.refresh()
    from kmcat import keyboardModeCAT
    pbar.update(1)

    pbar.set_description("Done")
    pbar.close()
