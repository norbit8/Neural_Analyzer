# -=-=-=-=-=-=--=-=-=-= IMPORTS =-=-=-=-=-=-=--
import pandas as pd
from scipy.ndimage import gaussian_filter
import fnmatch
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import random
import pickle
from scipy.io import loadmat
from typing import List
from graphs import *

LAST_COLUMN = -1


class decoder(object):
    """
    Decoder Class
    """
    NUMBER_OF_ITERATIONS = 100  # number of iteration of each group of cells for finding a solid average
    SIGMA = 30  # sigma for the gaussian
    NEIGHBORS = 1  # only closet neighbour, act like SVM
    TIMES = 30  # number of iteration on each K-population of cells.
    K = 48  # number of files per time
    LAG = 1000  # where to start the experiment (in the eye movement)
    d = {0: "PURSUIT", 1: "SACCADE"} # innder dictionary
    SEGMENTS = 12 #how many segment of 100ms we want to cut.
    SAMPLES_LOWER_BOUND = 100  # filter the cells with less than _ sampels
    number_of_cells_to_choose_for_test = 1 #when buildin X_test matrice, how many samples from each direction / reward
    step = 1

    def __init__(self, input_dir: str, output_dir: str, population_names: List[str]):
        """
        insert valid input_dir, output_dir and the population name mus be on
        @param input_dir:
        @param output_dir:
        @param population_names: must be from msn CRB ss cs SNR (mabye more..)
        """
        self.__input_dir = os.path.join(input_dir, '')
        self.__output_dir = os.path.join(output_dir, '')
        self.population_names = [x.upper() for x in population_names]
        self.__temp_path_for_writing = output_dir


    def filter_cells(self, cell_names, name):
        """
        remove from list the names which not conatin name string
        @param cell_names: list of the cell names
        @param name: SNR/msn/cs/.. etc
        @return:
        """
        return list(
            filter(lambda cell_name: True if cell_name.find(name) != -1 else False, [x.split(".")[0].upper() + "." + x.split(".")[1] for x in cell_names]))

    def convert_matlab_to_csv(self, exp="eyes", pop=0):
        """
        The expirement data is provided in the form of a MATLAB file, thus some pre-processing is needed
        in order to convert it to a more useable data-structre, in particular numpy array.
        Note that we convert the MATLAB file data to a pandas DataFrame and then we save it to a csv
        file for easier access in the future.
        @param exp:
        @param pop: 0 for pursuit 1 for saccade
        @return:
        """
        if (exp != "eyes" and exp !="rewards"):
            print("exp argument should be eyes or rewards only")
            return
        dir = self.__input_dir + exp + "/" + self.d[pop].lower() + "/"
        cell_names = fnmatch.filter(os.listdir(dir), '*.mat')  # filtering only the mat files.
        cell_names.sort()  # sorting the names of the files in order to create consistent runs.
        cells = []
        for name in self.population_names:
            cells += self.filter_cells(cell_names, name)
        for cell in cells:
            DATA_LOC = dir + cell  # cell file location
            data = loadmat(DATA_LOC)  # loading the matlab data file to dict
            if (exp == "eyes"):
                tg_dir = data['data']['target_direction'][0][0][0] / 45
            elif (exp == 'rewards'):
                tg_dir = data['data']['reward_probability'][0][0][0]
                tg_dir[tg_dir == 75] = 1
                tg_dir[tg_dir == 25] = 0
            spikes = data['data']['spikes'][0][0].todense().transpose()
            # tg_time = data['data']['target_motion'][0][0][0]
            mat = np.hstack([spikes, tg_dir.reshape(len(tg_dir), 1)])
            # saving the data to a csv file, and concatenating the number of samples from each file.
            if exp == "eyes":
                self.createDirectory("csv_files/eyes/" + self.d[pop] + "/")
            else:
                self.createDirectory("csv_files/rewards/" + self.d[pop] + "/")
            DataFrame(mat).to_csv(self.__temp_path_for_writing + str(spikes.shape[0]).upper() + "#" + cell[:-3] + "csv")

    def savesInfo(self, info, pop_type, expirience_type):
        """
        Saves the information of the trials into file
        @param info: the results to be saved
        @param pop_type: the name of the population SNR MSN etc..
        @param expirience_type: eyes or reward
        @return:
        """
        with open(self.__temp_path_for_writing + pop_type + expirience_type, 'wb') as info_file:
            pickle.dump(info, info_file)

    def saveToLogger(self, name_of_file_to_write_to_logger, type):
        """
        save to logger the populations the alorithm finished
        @param name_of_file_to_write_to_logger:
        @param type:
        @return:
        """
        with open(self.__temp_path_for_writing + "Logger" + self.d[type] + ".txt", "a+") as info_file:
            info_file.write(name_of_file_to_write_to_logger + "\n")

    def loadFromLogger(self, type):
        """
        load from logger all the population the logger already finished with
        @param type:
        @return:
        """
        try:
            l = []
            with open(self.__temp_path_for_writing + "Logger" + self.d[type] + ".txt", "r") as info_file:
                for line in info_file.readlines():
                    l.append(line.rstrip().split('_')[0])
            return l
        except:
            return []

    def filterWithGaussian(self, X):
        """
        Smoothing the Matrix of trials
        @param X: the matrice needed to be smooth
        @return:
        """
        for i in range(len(X)):
            X[i] = gaussian_filter(X[i], sigma=self.SIGMA)
        return X

    def extractNSampelsFromOneDirection(self, direction):
        """
        pick randomly x number of trials to test from one direction  when x = self.number_of_cells_to_choose_for_test
        @param direction:
        @return:
        """
        np.random.shuffle(direction)
        test = direction[:self.number_of_cells_to_choose_for_test]
        train = direction[self.number_of_cells_to_choose_for_test:]
        return train, test

    def SortMatriceToListOfDirections(self, X, y):
        """
        Given a matrix of neural spikes and the direction w.r.t each spike,
        generates list of bundled spikes which corresponds to the same direction in each bundle.
        each index of the list corresponds to the direction of the eye movement.
        Also the number of spikes (vectors) in each index of the list (directions) = n,
        which is the minimum number of directions from all the other choosen cells.
        The way we choose cells is explained in the main function.
        @param X:
        @param y:
        @return:
        """
        directions = []
        for i in range(int(np.amax(y)+1)):
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
        """
        split the X matrice into 2 matrices. one for the train and one for the test
        @param X:
        @param y:
        @return:
        """
        directions = self.SortMatriceToListOfDirections(X, y)
        averageVectorsMatrice, testSampelsMatrice = self.extractNSampelsFromAllDirections(directions)
        return averageVectorsMatrice, testSampelsMatrice

     #if type is eyes so type =8
    def getTestVectors(self, type=8):
        """
        creates the test and train vectors. we already know them without the X train and test matrice therefore we
        made them automatically. if the experiment is 'eyes' we know that there is 8 direction vectors
        @param type: 8 or 2 depending on the experiment (8 driections or 2 rewards type)
        @return:
        """
        y_train = np.hstack([i for i in range(type)]).flatten()
        y_test = np.array(sum([[j for i in range(self.number_of_cells_to_choose_for_test)] for j in range(type)], []))
        return y_train, y_test

    def mergeSampeling1(self, loadFromDisk):
        """
        makes one matrice from all the cell names from loadFromDist list
        @param loadFromDisk: the
        @return:
        """
        TrainAvgMatricesCombined = []
        testMatriceCombined = []
        for X, y in loadFromDisk:
            averageVectorsMatrice, testSampelsMatrice = self.createTrainAndTestMatrice(X, y)
            TrainAvgMatricesCombined.append(averageVectorsMatrice)
            testMatriceCombined.append(testSampelsMatrice)
        return np.hstack(TrainAvgMatricesCombined), np.hstack(testMatriceCombined)

    def readFromDisk(self, sampling, is_fragments=False, segment=0, EYES = True):
        """

        @param sampling: the names of the cells to read together and create one matrice
        @param is_fragments: to know if to split only the segmant or to read from 1000:2200
        @param segment:
        @param EYES: boolean - eyes or reward
        @return:
        """
        if (is_fragments):
            cut_first = self.LAG + (100 * segment)
            cut_last = self.LAG + (100 * (segment + 1))
        else:
            cut_first = self.LAG
            cut_last = LAST_COLUMN

        loadFiles = []
        for cell_name in sampling:
            dataset = pd.read_csv(self.temp_path_for_reading + cell_name)
            # print(dataset)
            X = dataset.iloc[:, cut_first: cut_last].values
            y = dataset.iloc[:, -1].values
            if EYES:
                X = self.filterWithGaussian(X)
            loadFiles.append((X, y))
        return loadFiles

    def filterCellsbyRows(self, cell_names):
        """
        filter the cells with lower bound of trials. if file is 148#SNR_4003 it means that this cell contain only 148
        trials
        @param cell_names:
        @return:
        """
        temp = []
        for cell_name in cell_names:
            new = cell_name[:cell_name.find("#")]
            if int(new) >= self.SAMPLES_LOWER_BOUND:
                temp.append(cell_name)
        return temp

    def control_group_cells(self, path):
        """
        inner function. check the simple knn algorithm validty.
        run only one cell each time and print the results
        path - absoult path os the folder containing the cells
        """
        self.temp_path_for_reading = path
        results = 0
        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(path), '*.csv')
        all_cell_names.sort()
        print(all_cell_names)
        classifier = KNeighborsClassifier(n_neighbors=self.NEIGHBORS, metric='minkowski', p=2, weights='distance')
        for cell in all_cell_names:
            # save the names of the cells and the score
            sum1 = 0
            # choose random K cells
            sampeling = [cell,]
            loadFiles = self.readFromDisk(sampeling)
            for i in range(self.NUMBER_OF_ITERATIONS):
                X_train, X_test = self.mergeSampeling1(loadFiles)
                y_train, y_test = self.getTestVectors()

                classifier.fit(X_train, y_train)
                y_pred2 = classifier.predict(X_test)
                sum1 += accuracy_score(y_test, y_pred2)
            print(cell, sum1 / self.NUMBER_OF_ITERATIONS)
            results += sum1 / self.NUMBER_OF_ITERATIONS
        print(results / len(all_cell_names))



    def simple_knn_eyes(self, type: int):
        """
        @param type: should be 0 for persuit or 1 for  saccade
        @return:
        """
        if (type not in [0, 1]):
            print("type should be 0 if pursuit or 1 is saccade")
            return

        self.temp_path_for_reading = self.__output_dir + "csv_files/eyes/" + self.d[type] + "/"
        self.createDirectory("EYES/" + self.d[type])

        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
        all_cell_names.sort()
        for population in [x for x in self.population_names if x not in self.loadFromLogger(type)]:
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
            for number_of_cells in range(1, self.K + 1, self.step):
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
                        y_train, y_test = self.getTestVectors(8)

                        classifier.fit(X_train, y_train)
                        y_pred2 = classifier.predict(X_test)
                        sum1 += accuracy_score(y_test, y_pred2)

                    totalAv += sum1 / self.NUMBER_OF_ITERATIONS
                    scoreForCells.append((sampeling, sum1 / self.NUMBER_OF_ITERATIONS))
                    infoPerGroupOfCells.append(scoreForCells)
                print(population, number_of_cells, totalAv / self.TIMES)
                info.append((infoPerGroupOfCells, totalAv / self.TIMES))
                sums.append(totalAv / self.TIMES)
            self.savesInfo(info, population, "")
            self.saveToLogger(population + "_" + self.d[type] + "_EYES", type)

    def createDirectory(self, name):
        if not os.path.exists(self.__output_dir + name):
            os.makedirs(self.__output_dir + name)
        self.__temp_path_for_writing = self.__output_dir + name + "/"

    def simple_knn_eye_fregment(self, type, choose_just_one=[], choose_of_segements=-1):
        """

        @param type:  should be 0 for persuit or 1 for  saccade
        @return:
        """
        self.temp_path_for_reading = self.__output_dir + "csv_files/eyes/" + self.d[type] + "/"

        self.createDirectory("EYES/" + self.d[type] + "_FRAGMENTS")

        if (type not in [0, 1]):
            print("type should be 0 if pursuit or 1 is saccade")
            return

        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
        all_cell_names.sort()
        iterate_population = self.population_names
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
            K = min(self.K,len(cell_names) - 1)
            start_choose_of_segements = 0

            if (choose_of_segements != -1):
                start_choose_of_segements = choose_of_segements

            for i in range(start_choose_of_segements, self.SEGMENTS):
                sums = []
                info = []
                segment = i
                for number_of_cells in range(1, K + 1, self.step):
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

    def file_namechanget(self, path):
        """
        helper func for name changing
        @param path:
        @return:
        """
        all_cell_names = fnmatch.filter(os.listdir(path), '*.mat')
        for name in all_cell_names:
            # print(name)

            #makes file name captial
            l = name.split('.')
            newName = l[0].upper() + "." + l[1]

            # reduce PC and BG in the begining
            # newName = name[3:]
            os.rename(path  + name, path + newName)

            # print(newName)


    def help(self):
        with open("decoder_instructions", 'r') as info_file:
            for line in info_file.readlines():
                print(line)


    def simple_knn_rewards(self, type: int):
        """
        @param type: should be 0 for persuit or 1 for  saccade
        @return:
        """

        if (type not in [0, 1]):
            print("type should be 0 if pursuit or 1 is saccade")
            return

        self.temp_path_for_reading = self.__output_dir + "csv_files/rewards/" + self.d[type] + "/"
        self.createDirectory("REWARDS/" + self.d[type])

        # loading folder
        all_cell_names = fnmatch.filter(os.listdir(self.temp_path_for_reading), '*.csv')
        all_cell_names.sort()
        for population in [x for x in self.population_names if x not in self.loadFromLogger(type)]:

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
            for number_of_cells in range(1, self.K + 1, self.step):
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
                        y_train, y_test = self.getTestVectors(type=2)
                        X_train = gaussian_filter(X_train, sigma = self.SIGMA)
                        classifier.fit(X_train, y_train)
                        y_pred2 = classifier.predict(X_test)
                        sum1 += accuracy_score(y_test, y_pred2)

                    totalAv += sum1 / self.NUMBER_OF_ITERATIONS
                    scoreForCells.append((sampeling, sum1 / self.NUMBER_OF_ITERATIONS))
                    infoPerGroupOfCells.append(scoreForCells)
                print(population, number_of_cells, totalAv / self.TIMES)
                info.append((infoPerGroupOfCells, totalAv / self.TIMES))
                sums.append(totalAv / self.TIMES)
            self.savesInfo(info, population, "")
            self.saveToLogger(population + "_" + self.d[type] + "_REWARDS", type)


## a.control_group_cells("/home/rachel/Neural_Analyzer/files/MATY_FILES/")

# a = decoder('/Users/shaigindin/MATY/Neural_Analyzer/files/in','/Users/shaigindin/MATY/Neural_Analyzer/files/out1',['SNR','msn','CRB','cs']
# a.convert_matlab_to_csv(exp="eyes", pop=0)
# a.convert_matlab_to_csv(exp="rewards", pop=1)


# a.simple_knn_eyes(type=0)
# a.simple_knn_eyes(1)

# a.simple_knn_rewards(0)
# a.simple_knn_rewards(1)
#
# a.simple_knn_eye_fregment(0)
# a.simple_knn_eye_fregment(1)


# a.simple_knn_eye_fregment(1)
# a.simple_knn_eyes(1)

# a.simple_knn_rewards(0)




g = Graphs(['SNR','msn','crb','cs'], ['pursuit','saccade'], '/Users/shaigindin/MATY/Neural_Analyzer/files/out/REWARDS/', fragments_cells=[0,4,7,9],load_fragments=False)
#
# g.plot_fragments()
# g.plot_experiments_same_populations()
g.plot_acc_over_concat_cells()
#



#
#
# dir =   "/home/rachel/Neural_Analyzer/files/in/rewards/pursuit/"
# all_cell_names = fnmatch.filter(os.listdir(dir), '*.mat')
#
# for name in all_cell_names:
#      # print(name)
#
#      #makes file name captial
#      l = name.split('.')
#      newName = l[0].upper() + "." + l[1]
#      #
#      #reduce PC and BG in the begining
#      # newName = name[3:]
#      os.rename(dir  + name, dir + newName)
# #
     # print(newName)