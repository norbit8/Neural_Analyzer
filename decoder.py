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
from scipy.io import loadmat
from pandas import DataFrame
from typing import List

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

    SAMPLES_LOWER_BOUND = 102  # filter the cells with less than _ sampels
    POPULATION_TYPE = "ss"  # type of population
    # DIR = '/content/drive/My Drive/mati_lab/final/csv_data/pursuit_8_dir_75and25/'

    # noga's cells  '/content/drive/My Drive/mati_lab/final/csv_data/pursuit_8_dir_75and25/'
    # mati's first good cells directory = '/content/drive/My Drive/mati_lab/first_assignment/code_and_data_2020/data/csv_files/'
    # mati's second good cells directory '/content/drive/My Drive/mati_lab/final/good_data/csv/'
    def __init__(self, input_dir: str, output_dir: str, population_names : List[str]):
        self._input_dir = input_dir # #note should be ende with / for example users/bin/
        self._output_dir = output_dir #note should be ende with / for example users/bin/
        self._population_names = population_names


    def filter_cells(self, cell_names, name):
        return list(filter(lambda cell_name: True if cell_name.find(name) != -1 else False, cell_names))


    def convert_matlab_to_csv(self, type='eyes'):
        """
        convert matlab files to csv files. each file represent a table of the trials and last coloum will be the
        type. if you want to analyze eyes movement type=eyes etc..
        """
        DATA_DIR = self._input_dir
        cell_names = fnmatch.filter(os.listdir(DATA_DIR), '*.mat')  # filtering only the mat files.
        cell_names.sort()  # sorting the names of the files in order to create consistent runs.
        cells = []
        for name in self._population_names:
           cells += self.filter_cells(cell_names, name)
        cell_names = cells
        for cell in cell_names:
            DATA_LOC = DATA_DIR + cell  # cell file location
            data = loadmat(DATA_LOC)  # loading the matlab data file to dict
            if (type == 'eyes'):
                tg_dir = data['data']['target_direction'][0][0][0] / 45
            elif (type == 'rewards'):
                tg_dir = data['data']['reward_probability'][0][0][0]
                tg_dir[tg_dir == 75] = 1
                tg_dir[tg_dir == 25] = 0
            spikes = data['data']['spikes'][0][0].todense().transpose()
            tg_time = data['data']['target_motion'][0][0][0]
            mat = np.hstack([spikes, tg_dir.reshape(len(tg_dir), 1)])
            # saving the data to a csv file, and concatenating the number of samples from each file.
            DataFrame(mat).to_csv(self._output_dir + str(spikes.shape[0]) + "#" + cell[:-3] + "csv")



    def algo(self):
        pass