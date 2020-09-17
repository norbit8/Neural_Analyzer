import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import resample
from scipy.ndimage import gaussian_filter
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import fnmatch
import os
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.svm import SVC
import pickle

class decoder(object):
    """
    Decoder Class
    """
    NUMBER_OF_ITERATIONS = 50  # number of iteration of each group of cells for finding a solid average
    SIGMA = 30  # sigma for the gaussian
    NEIGHBORS = 1  # only closet neighbour, act like SVM
    TIMES = 50  # number of iteration on each K-population of cells.
    K = 48  # number of files per time
    n = 15  # how many rows to poll out of around 17-20 rows

    SAMPLES_LOWER_BOUND = 101  # filter the cells with less than _ sampels
    POPULATION_TYPE = "ss"  # type of population
    DIR = '/content/drive/My Drive/mati_lab/final/csv_data/pursuit_8_dir_75and25/'

    # noga's cells  '/content/drive/My Drive/mati_lab/final/csv_data/pursuit_8_dir_75and25/'
    # mati's first good cells directory = '/content/drive/My Drive/mati_lab/first_assignment/code_and_data_2020/data/csv_files/'
    # mati's second good cells directory '/content/drive/My Drive/mati_lab/final/good_data/csv/'
    def __init__(self, input_dir: str, output_dir: str):
        self._input_dir = input_dir
        self._output_dir = output_dir


    def convert_matlab_to_csv(self):
        pass
