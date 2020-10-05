# IMPORTS
from typing import List
import pickle
import os
import numpy as np
import math
from pandas import DataFrame
from plotnine import *  # ggplot

from wcmatch import fnmatch as fn
from decoder import *

CRED = '\033[91m'  # RED COLOR
CGREEN = '\033[92m'  # GREEN COLOR
CEND = '\033[0m'  # END WITH COLORS


class Graphs:
    """
    Graphs class.
    Responsible for drawing and getting all of the data generated by the decoder class.
    """

    def __init__(self):
        """
        Ctor of the graphs class
        """
        pass

    # @staticmethod
    # def get_population_one_cell_data_frame(file_path: str):
    #     cell_names = []
    #     if os.path.isdir(file_path):
    #         cell_names = fnmatch.filter(os.listdir(file_path), '*')
    #         cell_names = fn.filter(cell_names, ALL_POSSIBILE_POPULATIONS)
    #         file_path = os.path.join(file_path, '')
    #         cell_names = [file_path + name for name in cell_names]
    #     elif os.path.isfile(file_path):
    #         cell_names = [file_path]
    #     names_list = []
    #     rate_list = []
    #     population_list = []
    #     for file_name_path in cell_names:
    #         with open(file_name_path, 'rb') as info_file:
    #             info = pickle.load(info_file)
    #             for tup in info[0][0]:
    #                 names_list.append(decoder.get_cell_name(tup[0]))
    #                 rate_list.append(tup[1])
    #                 population_list.append(decoder.get_population_name(tup[0]))
    #     return DataFrame({'cell_name': names_list, 'acc': rate_list, 'type': population_list})

    # @staticmethod
    # def get_acc_df_for_graph(file_paths: List):
    #     algo_name_list = []
    #     kind_name_list = []
    #     rate_list = []
    #     population_name_list = []
    #     K_population = []
    #     expirement_list = []
    #     group = []
    #     for file_path in file_paths:
    #         if os.path.isdir(file_path):
    #             cell_names = fnmatch.filter(os.listdir(file_path), '*')
    #             cell_names = fn.filter(cell_names, ALL_POSSIBILE_POPULATIONS)
    #             file_path = os.path.join(file_path, '')
    #             cell_names = [file_path + name for name in cell_names]
    #         elif os.path.isfile(file_path):
    #             cell_names = [file_path]
    #         else:
    #             print("file path is not valid")
    #             exit(1)
    #         for file_name_path in cell_names:
    #             with open(file_name_path, 'rb') as info_file:
    #                 # print("CHECK: ", Graphs.create_std_data_frag(info_file))
    #                 info = pickle.load(info_file)
    #                 for i, tup in enumerate(info):
    #                     if i == 0:
    #                         K_population.append(i + 1)
    #                     else:
    #                         K_population.append(len(tup[0][0][0][0]))
    #                     print("tup: ", tup)
    #                     rate_list.append(np.around(tup[1], 3))
    #                     name = os.path.basename(file_name_path)
    #                     population_name_list.append(name)
    #                     algo_name = os.path.basename(os.path.dirname(file_name_path))
    #                     algo_name_list.append(algo_name)
    #                     kind_name = os.path.basename(os.path.dirname(os.path.dirname(file_name_path)))
    #                     kind_name_list.append(kind_name)
    #                     expirement_name = os.path.basename(
    #                         os.path.dirname(os.path.dirname(os.path.dirname(file_name_path))))
    #                     expirement_list.append(os.path.basename(expirement_name))
    #                     group.append("\n".join([expirement_name, kind_name, algo_name, name]))
    #     return DataFrame({'concatenated_cells': K_population, 'acc': rate_list,
    #                       'population': population_name_list, 'kind': kind_name_list, 'algorithm': algo_name_list,
    #                       'experiment': expirement_list, 'group': group})

    @staticmethod
    def plot_acc_over_concat_cells(files: List[str] = None, number_of_concat_cells: int = 100, ticks: int = 1):
        """
        Drawing a graphs of accuracies over number of concatenated cells, of all the populations selected.
        and of all of the experiment types.
        :param populations: Selected populations.
        :param experiment_types: experiment types.
        :param number_of_concat_cells: Number of cells to show !!!INCLUSIVE!!!. (X Axis ticks)
        :return: The filtered DataFrame.
        """
        filtered_dataframe = decoder.get_acc_df_for_graph(files)
        print(ggplot(data=filtered_dataframe,
                     mapping=aes(x='concatenated_cells', y='acc', color='group', group='group')) + \
              geom_line() +
              geom_point() +
              scale_x_continuous(
                  breaks=np.arange(1, number_of_concat_cells + 1, ticks)) +  # X axis leaving out some ticks
              scale_y_continuous(breaks=np.arange(0, 1, 0.05)) +  # Y axis scaling
              labs(x='Number of concatenated cells', y='Accuracy') +
              theme_classic())
        return filtered_dataframe

    @staticmethod
    def data_acc_over_concat_cells(files: List[str] = None):
        """
        Gets the data frame of accuracies over number of concatenated cells, of all the populations selected.
        and of all of the experiment types.
        :param files - list of files.
        :return: The filtered DataFrame.
        """
        if files is None:
            return None
        return decoder.get_acc_df_for_graph(files)

    @staticmethod
    def plot_histogram(files: List[str] = None):
        new_arr = []
        steps = np.arange(0, 1, 0.2)
        for step in steps:
            new_arr.append(str(int(step * 100)) + "% - " + str(int((step + 0.2) * 100)) + "%")

        for file in files:
            if os.path.isdir(file):
                if os.name == 'nt':  # windows
                    file = file + "\\"
                elif os.name == 'posix':  # Linux , Mac OSX and all the other posix compatible os.
                    file = file + "/"
                else:
                    file = file + "/"
                all_the_files = [os.path.dirname(file) + "/" + file2 for file2 in os.listdir(file)]
                all_the_files = [name for name in all_the_files if os.path.basename(name) in ALL_POSSIBILE_POPULATIONS]

            else:
                all_the_files = [file]
            for file2 in all_the_files:
                acc = []
                df = decoder.get_population_one_cell_data_frame(file2)
                total_num = df.shape[0]
                acc.append(df.loc[(df['acc'] >= 0) & (df['acc'] <= 0 + 0.2)].shape[0])
                acc.extend([df.loc[(df['acc'] > x) & (df['acc'] <= x + 0.2)].shape[0] for x in steps[1:]])
                final_data = DataFrame({'acc': new_arr, 'cell_percentage': [x / total_num for x in acc]})
                print(ggplot(data=final_data,
                             mapping=aes(x='acc', y='cell_percentage')) + \
                      geom_bar(stat="identity") +
                      ggtitle(os.path.basename(file2)) +
                      labs(x='Accuracy', y='Cell percentage') +
                      theme_classic())

    def creates_data_frag(self, info):
        success_rate = []
        number_of_cells = []
        CRB = []
        for i, k in enumerate(info):
            number_of_cells.append(i + 1)
            success_rate.append(k[1])
        CRB.append(number_of_cells)
        CRB.append(success_rate)
        return CRB

    @staticmethod
    def create_std_data_frag(info):
        """
        creates the std
        :param info:
        :return:
        """
        stds = []
        number_of_cells = []
        for i, k in enumerate(info):
            number_of_cells.append(i + 1)
            stds.append(
                np.std(np.array([score_for_cells[0][1] for score_for_cells in k[0]]), ddof=1) / math.sqrt(50))
        stds_per_files = (number_of_cells, stds)
        return stds_per_files

    def data_loader_fragments(self, fragments_cells: List[int]):
        """
        Creating the fragments list
        :param number_of_fragments: number of fragments.
        :return: list of the loaded data.
        """
        # REG
        all = []
        for index, exp in enumerate(self.__kind_names):
            all.append([])
            for pop in self.__populations:
                temp = []
                for i in range(12):
                    with open(self.__base_dir + exp.upper() + '_FRAGMENTS' + self.__slash +
                              str.upper(pop) + str(i), 'rb') as info_file:
                        info = pickle.load(info_file)
                        temp.append(self.creates_data_frag(info))
                all[index].append(temp)
        # STD
        all2 = []
        for index, exp in enumerate(self.__kind_names):
            all2.append([])
            for pop in self.__populations:
                temp2 = []
                for i in range(12):
                    with open(self.__base_dir + exp.upper() + '_FRAGMENTS' + self.__slash +
                              str.upper(pop) + str(i), 'rb') as info_file:
                        info = pickle.load(info_file)
                        temp2.append(self.create_std_data_frag(info))
                all2[index].append(temp2)
        # loading the df
        self.__frag_df = DataFrame(
            columns=['time', 'acc', 'population', 'concatenated_cells', 'experiment_type', 'std'])
        for time in range(12):  # Time
            for index, exp_type in enumerate(self.__kind_names):  # Saccade vs pursuit
                for pop_index, population_type in enumerate(self.__populations):  # Population
                    # for cell_number in [0, 9, 19, 29]:  # Cell numbers
                    for cell_number in fragments_cells:  # Cell numbers
                        try:  # Only because of the Saccade ss problem
                            # (time * 100) + 1000
                            self.__frag_df = self.__frag_df.append({'time': time,
                                                                    'acc': all[index][pop_index][time][1][cell_number],
                                                                    'population': population_type,
                                                                    'concatenated_cells': cell_number + 1,
                                                                    'experiment_type': exp_type,
                                                                    'std': all2[index][pop_index][time][1][
                                                                        cell_number]}, ignore_index=True)
                        except:
                            print("ERROR: we don't have " + str(cell_number) + " cells in:",
                                  exp_type, population_type + '.', '(Time: ' + str(time) + ')')

    def get_fragments(self, populations: List[str] = None,
                      experiments: List[str] = None):
        """
        Getter for the fragments data
        :return: fragments data frame
        """
        if self.__frag_df is None:
            print(CRED + "Fragments are not loaded, please create a new graphs instance with the relevant flag" + CEND)
            return
        if populations is None:
            populations = self.__populations
        if experiments is None:
            experiments = self.__kind_names
        return self.__frag_df[(self.__frag_df['population'].isin(populations)) &
                              (self.__frag_df['experiment_type'].isin(experiments))]

    def plot_fragments(self, populations: List[str] = None,
                       experiments: List[str] = None):
        """
        Printing the fragments graphs
        :return: None if there is an error, and the DataFrame representing the fragments data.
        """
        if self.__frag_df is None:
            print(CRED + "Fragments are not loaded, please create a new graphs instance with the relevant flag" + CEND)
            return
        if populations is None:
            populations = self.__populations
        if experiments is None:
            experiments = self.__kind_names
        for exp_index, exp_type in enumerate(experiments):
            print("Printing " + exp_type)
            for population in populations:
                print(ggplot(data=self.__frag_df[(self.__frag_df['population'] == population) &
                                                 (self.__frag_df['experiment_type'] == exp_type)],
                             mapping=aes(x='time', y='acc', group=1)) +
                      geom_line(color='red') +
                      geom_point() + facet_wrap('~concatenated_cells') +
                      ggtitle(population + ' ' + exp_type) +
                      theme_classic() +
                      geom_errorbar(mapping=aes(x="time", ymin='acc-std', ymax='acc+std')))
        return self.__frag_df

    @staticmethod
    def plot_acurracy_comparision(one:str, two:str):
        try:
            df1 = decoder.get_population_one_cell_data_frame(one)
            df2 = decoder.get_population_one_cell_data_frame(two)
        except:
            print("input is invalid: one (or both) of the files is not ok")
            exit(0)
        df1.rename(columns={"acc": "acc1"}, inplace=True)
        df2.rename(columns={"acc": "acc2"}, inplace=True)
        final_df = pd.merge(df1, df2, how ='inner', on =['cell_name'])
        if final_df.shape[0] == 0:
            print("no matching cells")
            exit(0)
        print(ggplot(data=final_df,
                     mapping=aes(x="acc1", y="acc2",color="cell_name")) + \
        geom_point(alpha=0.8) + \
              geom_text(aes(label="cell_name"), fontweight="5") +\
        labs(x=decoder.get_full_name(one) + ' Accuracy', y=decoder.get_full_name(two) + ',Accuracy') + \
              scale_x_continuous(breaks=np.arange(0,  1, 0.05)) + \
              scale_y_continuous(breaks=np.arange(0, 1, 0.05))
        )
        return final_df

    def help(self):
        print("commands: \n"
              " (*) plot_acc_over_concat_cells(populations: List[str] = None,"
              "experiment_types: List[str] = None, number_of_concat_cells: int = 100,"
              "ticks: int = 1"
              ") - Draws the graph of the concatenated cells over their accuracies.\n"
              " USAGE: \n"
              "populations - Provide a list of all of the desired populations. example: ['SNR', 'CRB', 'SS'] \n"
              "experiment_types - Provide a list of all the experiment types. example: ['PURSUIT', 'SACCADE']\n"
              "number_of_concat_cells - Number of concatenated cells to show. example: 30\n"
              "ticks - The jumps on the X axis. example: 1 \n\n"
              " (*) data_acc_over_concat_cells() - Getter for the data of the concatenated cells "
              "over their accuracies (Pandas DataFrame)\n"
              "USAGE: \n"
              "populations - Provide a list of all of the desired populations. example: ['SNR', 'CRB', 'SS'] \n"
              "experiment_types - Provide a list of all the experiment types. example: ['PURSUIT', 'SACCADE']\n"
              "number_of_concat_cells - Number of concatenated cells to show. example: 30\n\n"
              " (*) plot_fragments() - Plots the fragments.\n\n"
              " (*) get_fragments() - Getter for the fragments data. (Pandas DataFrame)")


#####################################################################
#                       FOR THE MEETING
#####################################################################
## EXAMPLES ##
# from graphs import *
# g = Graphs(['CRB', 'SS','CS','MSN', 'SNR'],['PURSUIT','SACCADE'], '/home/mercydude/Desktop/Neural_Analyzer/files/out/EYES', load_fragments=False)
# g.plot_acc_over_concat_cells(ticks=5)  # argument ticks
# g.plot_experiments_same_populations()

## FRAGMENTS ##
# g = Graphs(['CRB','SS','MSN','SNR'],['PURSUIT','SACCADE'], '/home/mercydude/Desktop/Neural_Analyzer/files/out/EYES', load_fragments=True)
#
# g.plot_fragments()

# g.help()

## GETTERS:

# g.get_fragments()
# g.data_acc_over_concat_cells()
#
#####################################################################
#                               END
#####################################################################
# folder = "/Users/shaigindin/MATY/Neural_Analyzer/files/out1/project_name/target_direction/pursuit/simple_knn/"
# pursuit_folder = "/Users/shaigindin/MATY/Neural_Analyzer/noga_out/project_name/target_direction/pursuit/simple_knn/SNR"
# saccade_folder = "/Users/shaigindin/MATY/Neural_Analyzer/noga_out/project_name/target_direction/saccade/simple_knn/SNR"
# Graphs.plot_acc_over_concat_cells([pursuit_folder])
# Graphs.plot_acc_over_concat_cells([saccade_folder])
# Graphs.plot_histogram([pursuit_folder])
# Graphs.plot_acurracy_comparision(pursuit_folder, saccade_folder)