# -=-=-=-=-=-=--=-=-=-= IMPORTS =-=-=-=-=-=-=--
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

LAST_COLUMN = -1


class decoder(object):
    """
    Decoder Class
    """
    NUMBER_OF_ITERATIONS = 1  # number of iteration of each group of cells for finding a solid average
    SIGMA = 30  # sigma for the gaussian
    NEIGHBORS = 1  # only closet neighbour, act like SVM
    TIMES = 1  # number of iteration on each K-population of cells.
    K = 48  # number of files per time default
    LAG = 1000  # in ms
    d = {0: "PURSUIT", 1: "SACCADE"}
    SAMPLES_LOWER_BOUND = 10  # filter the cells with less than _ sampels
    number_of_cells_to_choose_for_test = 1
    SEGMENTS = 12

    def __init__(self, input_dir: str, output_dir: str, population_names: List[str]):
        """
        insert valid input_dir, output_dir and the population name mus be on
        @param input_dir:
        @param output_dir:
        @param population_names: must be from msn CRB ss cs SNR (mabye more..)
        """
        self._input_dir = os.path.join(input_dir, '')
        self._output_dir = os.path.join(output_dir, '')
        self.temp_path_for_writing = output_dir
        self._population_names = [x.upper() for x in population_names]

    def filter_cells(self, cell_names, name):
        return list(
            filter(lambda cell_name: True if cell_name.find(name) != -1 else False, [x.upper() for x in cell_names]))

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
            self.createDirectory("csv_files")
            DataFrame(mat).to_csv(self.temp_path_for_writing + str(spikes.shape[0]).upper() + "#" + cell[:-3] + "csv")

    def savesInfo(self, info, pop_type, expirience_type):
        with open(self.temp_path_for_writing + pop_type + "_" + expirience_type, 'wb') as info_file:
            pickle.dump(info, info_file)

    def saveToLogger(self, name_of_file_to_write_to_logger, type):
        with open(self.temp_path_for_writing + "Logger" + self.d[type] + ".txt", "a+") as info_file:
            info_file.write(name_of_file_to_write_to_logger + "\n")

    def loadFromLogger(self, type):
        try:
            l = []
            with open(self.temp_path_for_writing + "Logger" + self.d[type] + ".txt", "r") as info_file:
                for line in info_file.readlines():
                    l.append(line.rstrip().split('_')[0])
            return l
        except:
            return []

    def filterWithGaussian(self, X):
        for i in range(len(X)):
            X[i] = gaussian_filter(X[i], sigma=self.SIGMA)
        return X

    def extractNSampelsFromOneDirection(self, direction):
        np.random.shuffle(direction)
        test = direction[:self.number_of_cells_to_choose_for_test]
        train = direction[self.number_of_cells_to_choose_for_test:]
        return train, test

    def SortMatriceToListOfDirections(self, X, y):
        directions = []
        for i in range(8):
            idx = y == i
            temp = X[idx, :]
            directions.append(temp)
        return directions

    def extractNSampelsFromAllDirections(self, directions):
        directionsAverageVector = []
        testSampels = []
        for direction in directions:
            train, test = self.extractNSampelsFromOneDirection(direction)
            testSampels.append(test)
            averageVector = np.sum(np.array(train), axis=0) / train.shape[0]
            directionsAverageVector.append(averageVector)
        return np.vstack(directionsAverageVector), np.vstack(testSampels)

    def createTrainAndTestMatrice(self, X, y):
        directions = self.SortMatriceToListOfDirections(X, y)
        averageVectorsMatrice, testSampelsMatrice = self.extractNSampelsFromAllDirections(directions)
        return averageVectorsMatrice, testSampelsMatrice

    def getTestVectors(self):
        y_train = np.hstack([i for i in range(8)]).flatten()
        y_test = np.array(sum([[j for i in range(self.number_of_cells_to_choose_for_test)] for j in range(8)], []))
        return y_train, y_test

    def mergeSampeling1(self, loadFromDisk):
        TrainAvgMatricesCombined = []
        testMatriceCombined = []
        for X, y in loadFromDisk:
            averageVectorsMatrice, testSampelsMatrice = self.createTrainAndTestMatrice(X, y)
            TrainAvgMatricesCombined.append(averageVectorsMatrice)
            testMatriceCombined.append(testSampelsMatrice)
        return np.hstack(TrainAvgMatricesCombined), np.hstack(testMatriceCombined)

    def readFromDisk(self, sampling, is_fragments=False, segment=0):
        if (is_fragments):
            cut_first = self.LAG + 100 * segment
            cut_last = self.LAG + 100 * (segment + 1)
        else:
            cut_first = self.LAG
            cut_last = LAST_COLUMN
        loadFiles = []
        for cell_name in sampling:
            dataset = pd.read_csv(self.temp_path_for_reading + cell_name)
            X = dataset.iloc[:, cut_first: cut_last].values
            y = dataset.iloc[:, -1].values
            X = self.filterWithGaussian(X)
            loadFiles.append((X, y))
        return loadFiles

    def filterCellsbyRows(self, cell_names):
        temp = []
        for cell_name in cell_names:
            new = cell_name[:cell_name.find("#")]
            if int(new) >= self.SAMPLES_LOWER_BOUND:
                temp.append(cell_name)
        return temp

    def simple_knn_eyes(self, type: int):
        """

        @param type: should be 0 for persuit or 1 for  saccade
        @return:
        """

        if (type not in [0, 1]):
            print("type should be 0 if pursuit or 1 is saccade")
            return

        self.temp_path_for_reading = self._output_dir + "csv_files/"
        self.createDirectory(self.d[type])
        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(self._input_dir), '*.csv')
        all_cell_names.sort()
        for population in [x for x in self._population_names if x not in self.loadFromLogger(type)]:
            cell_names = self.filter_cells(all_cell_names, population)
            cell_names = self.filterCellsbyRows(cell_names)

            # build list which saves info
            info = []

            if (self.K > len(cell_names) - 1):
                self.K = len(cell_names) - 1

            # saves the rate of the success for each k population
            sums = []
            classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')
            # iterating over k-population of cells from 1 to K
            for number_of_cells in range(1, self.K + 1):
                # saves each groupCells
                infoPerGroupOfCells = []

                # intializing counter
                totalAv = 0

                # iterating TImes for solid average
                for j in range(self.TIMES):
                    # save the names of the cells and the score
                    scoreForCells = []

                    sum1 = 0
                    # choose random K cells
                    sampeling = random.sample(cell_names, k=number_of_cells)
                    loadFiles = self.readFromDisk(sampeling)
                    for i in range(self.NUMBER_OF_ITERATIONS):
                        X_train, X_test = self.mergeSampeling1(loadFiles)
                        y_train, y_test = self.getTestVectors()

                        classifier.fit(X_train, y_train)
                        y_pred2 = classifier.predict(X_test)
                        sum1 += accuracy_score(y_test, y_pred2)

                    totalAv += sum1 / self.NUMBER_OF_ITERATIONS
                    scoreForCells.append((sampeling, sum1 / self.NUMBER_OF_ITERATIONS))
                    infoPerGroupOfCells.append(scoreForCells)
                info.append((infoPerGroupOfCells, totalAv / self.TIMES))
                sums.append(totalAv / self.TIMES)
            self.savesInfo(info, population, self.d[type] + "_EYES")
            self.saveToLogger(population + "_" + self.d[type] + "_EYES", type)

    def createDirectory(self, name):
        if not os.path.exists(self._output_dir + name):
            os.makedirs(self._output_dir + name)
        self.temp_path_for_writing = self._output_dir + name + "/"

    def simple_knn_eye_fregment(self, type, choose_just_one=[], choose_of_segements=-1):
        """

        @param type:  should be 0 for persuit or 1 for  saccade
        @return:
        """
        self.temp_path_for_reading = self._output_dir + "csv_files/"

        self.createDirectory(self.d[type] + "_FRAGMENTS")

        if (type not in [0, 1]):
            print("type should be 0 if pursuit or 1 is saccade")
            return

        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
        all_cell_names.sort()
        iterate_population = self._population_names
        if choose_just_one != []:
            if len(choose_just_one) != 1:
                print("must be just one population e.g [msn,]")
                return
            else:
                iterate_population = choose_just_one
        for population in iterate_population:
            POPULATION_TYPE = population

            # create classifier
            classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')

            cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
            cell_names.sort()

            cell_names = self.filter_cells(cell_names, POPULATION_TYPE)
            cell_names = self.filterCellsbyRows(cell_names)
            # limit K to 48 or popultaion (lower bound)
            K = K if len(cell_names) - 1 > 50 else len(cell_names) - 1

            start_choose_of_segements = 0
            if (choose_of_segements != -1):
                start_choose_of_segements = choose_of_segements

            for i in range(start_choose_of_segements, self.SEGMENTS):
                sums = []
                info = []
                segment = i
                for number_of_cells in range(1, K + 1):
                    # saves each groupCells
                    infoPerGroupOfCells = []

                    # intializing counter
                    totalAv = 0
                    for j in range(self.TIMES):
                        # save the names of the cells and the score
                        scoreForCells = []
                        sum = 0
                        # choose random K cells
                        sampeling = random.sample(cell_names, k=number_of_cells)
                        loadFiles = self.readFromDisk(sampeling, is_fragments=True, segment=segment)
                        for i in range(self.NUMBER_OF_ITERATIONS):
                            X_train, X_test = self.mergeSampeling1(loadFiles)
                            y_train, y_test = self.getTestVectors()
                            classifier.fit(X_train, y_train)
                            y_pred2 = classifier.predict(X_test)
                            # np.random.shuffle(y_test)
                            sum += accuracy_score(y_test, y_pred2)
                        totalAv += sum / self.NUMBER_OF_ITERATIONS
                        scoreForCells.append((sampeling, sum / self.NUMBER_OF_ITERATIONS))
                        infoPerGroupOfCells.append(scoreForCells)
                    info.append((infoPerGroupOfCells, totalAv / self.TIMES))
                    sums.append(totalAv / self.TIMES)
                    self.savesInfo(info, population,  str(segment))
            self.saveToLogger(population + "_" + self.d[type] + "_EYES", type)



a = decoder('/Users/shaigindin/MATY/Neural_Analyzer/files/','/Users/shaigindin/MATY/Neural_Analyzer/files/out/', ['SNR','msn','CRB'])
a.simple_knn_eye_fregment(0)
a.simple_knn_eye_fregment(1)