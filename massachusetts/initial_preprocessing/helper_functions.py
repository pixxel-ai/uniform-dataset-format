import numpy as np
import cv2
from pathlib import Path
from fastai.vision import open_mask

def combine_masks(road, building):
    """
    Combines a Road and a building mask
    Each mask has to be a numpy array of 2 dimensions containing only two unique values:
    1. Background = 0
    2. Element (road, building, etc) = INTEGER
    Assumes:
    # Roads will be stored with value 1 in mask
    # Buildings will be stored with value 2 in mask
    # Roads take precedence where both roads and buildings intersect
    """
    road = road.clip(0, 1)
    building = building.clip(0, 1) * 2
    building = building - building * road
    return road + building

def save_mask(mask, dest, ext='.png'):
    """
    `mask` : Takes as input a 2 dimensional numpy array as a Mask
    `dest` : Should be of type : `str` or `pathlib.Path`
    `ext`  : Extension for mask to be saved
    The function takes the mask and saves it in the destination with extension `ext`
    
    """
    dest = Path(dest)
    dest = dest.parent/(dest.stem + ext)
    cv2.imwrite(str(dest), mask)
    
def get_masks(f):
    """
    A function specifically made for the massachusetts dataset
    Checks if a particular image has both a roads and a buildings mask and returns Either of both of them
    """
    r_f = R_MSKS/f  # road file name
    b_f = B_MSKS/f  # building file name
    road = open_mask(r_f).data.squeeze().numpy()
    building = 'False'   # to indicate that a mask has not been read yet
    if b_f.is_file():
        building = open_mask(b_f).data.squeeze().numpy()
    return road, building
    