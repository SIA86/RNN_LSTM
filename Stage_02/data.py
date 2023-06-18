import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import mplfinance as mpf
from scipy.signal import argrelextrema
import sys, os
from typing import Tuple

class Analyzer():
    """
    Помогает проанализировать полученные от НС данные. 
    - высчитывает среднюю ошибку предсказаний 'Loss'
    - высчитывает величину изменения цены на основании предсказаний 'Pred_Close_diff' 
    - проверяет совпадение предсказанного направления движения цены с фактическим 'MAtch'
    - позволяет отфильтровать результаты по колонкам
    """
    def __init__(self, path):
        self.path = path
        self.data = self.__get_data()

        self.filtered = None
        self.column_name = None
        self.upper = None
        self.down = None
        self.inner = None
        

    def __get_data(self):
        df = pd.read_csv(self.path, index_col=0)
        df = df[['Close', 'Predicted_close']]
        df['Pred_Close_diff'] = df['Predicted_close'] -df['Close'] 
        df['Fact_diff'] = df['Close'].diff().shift(-1)
        df = df.dropna()
        df['Loss'] = abs(df['Pred_Close_diff']-df['Fact_diff'])

        df.loc[((df['Pred_Close_diff'] >= 0) & (df['Fact_diff'] >= 0)) | ((df['Pred_Close_diff'] < 0) & (df['Fact_diff'] < 0)), 'Match'] = 1 
        df.loc[((df['Pred_Close_diff'] >= 0) & (df['Fact_diff'] < 0)) | ((df['Pred_Close_diff'] < 0) & (df['Fact_diff'] >= 0)), 'Match'] = 0

        return df

    def filtration(self): 
        self.column_name = 'Pred_Close_diff'
        self.upper = int(self.data['Loss'].mean())
        self.down = -int(self.data['Loss'].mean())
        self.filtered = self.data.copy()
        self.filtered = self.filtered.loc[(self.data['Pred_Close_diff'] < self.down) | (self.data['Pred_Close_diff'] > self.upper)]

    def custom_filtration(self, column_name : str = 'Pred_Close_diff', upper: int = None, down: int = None, inner: bool = False):
        self.upper = upper
        self.down = down
        self.column_name = column_name
        self.inner = inner
        self.filtered = self.data.copy()
        if self.inner:
            if self.upper and self.down:
                self.filtered = self.filtered.loc[(self.data[column_name] <= upper) & (self.data['Pred_Close_diff'] >= down)]
            elif self.upper:
                self.filtered = self.filtered.loc[(self.data[column_name] <= upper)]
            else:
                self.filtered = self.filtered.loc[(self.data[column_name] >= down)]
        else:
            self.filtered = self.filtered.loc[(self.data[column_name] > upper) | (self.data['Pred_Close_diff'] < down)]
                
        
    def __str__(self):        
        info = '\n'.join([
            f"{self.path.removeprefix(f'data{os.sep}predictions{os.sep}').removesuffix('.csv')} analyzing:",
            f"Mean loss (pt): {int(self.data['Loss'].mean())}",
            f"Matching ratio: {int(self.data['Match'].value_counts()[1]/(self.data['Match'].value_counts()[1]+self.data['Match'].value_counts()[0])*100)}%"
        ])

        if type(self.filtered).__name__ != 'NoneType':
            info = info + '\n'.join([
                f"\nData filtered by column ['{self.column_name}']",
                f"{f'Condition: {self.upper} > values > {self.down}' if self.inner else f'Condition: value > {self.upper} | value < {self.down}'}", 
                f"Data compression: {100 - int(self.filtered.shape[0]/self.data.shape[0]*100)}%",
                f"Filtered matching ratio: {int(self.filtered['Match'].value_counts()[1]/(self.filtered['Match'].value_counts()[1]+self.filtered['Match'].value_counts()[0])*100)}%"
            ])
        else:
            info = info + f"\nNo filtered data"
            
        return info


class Utils():
    """
    Набор полезных утилит для работы с данными.
    
    split data - разбивает базу данных на части (train, val, test) для последующей передачи нейронной сети
    data_divider - помогает разбить данные на части
    """
    @staticmethod
    def split_data(data : pd.DataFrame | np.ndarray, 
                split: float, val : float | None = None
                ) -> Tuple[pd.DataFrame, ...] | Tuple[np.ndarray, ...]:

        size = int(len(data)*split)
        spl_test = data[size:]
        if val:
            val_size = int(size*(1-val))       
            spl_train = data[:val_size]
            spl_val = data[val_size:size]
        else:
            spl_train = data[:size]
            spl_val = None

        return spl_train, spl_val, spl_test
    
    @staticmethod
    def data_divider(path : str, length : int, shift : int = 0, start: int = 0):
        filepath_dir = os.path.dirname(path)
        with open(path, 'r') as file_reader:
            header = file_reader.readline()
            data = file_reader.readlines()
        if shift:
            for pos in range(start, len(data)-length, shift):
                with open(f'{filepath_dir}{os.sep}data_{pos}-{length+pos}.csv', 'w') as fw:
                    fw.write(header)
                    fw.writelines(data[pos:length+pos])
        else:
            with open(f'{filepath_dir}{os.sep}data_{start}-{length+start}.csv', 'w') as fw:
                    fw.write(header)
                    fw.writelines(data[start:length+start])



class ProcessData():
    def __init__(self, path: str, accuracy : int = 5,  norm_algorythm : str | None = None):   
        """ 
        Creating dataframe object from *.csv file getting from finam.ru stock prices archive 
        and processing data for further dataset creation.

        Parametrs:
        -----------
        path : str
            relative path to csv data file
        accuracy : int
            rounded data values to ndigits precision after the decimal point
        
        Returns:
        ----------
            data - pd.DataFrame raw data
        """                            
        self.path = path
        self.accuracy = accuracy
        self.plot_info = {
            'subplot':[],
            'plot':[]
        }
        self.__data = self.__get_data()
        

    def __get_data(self):
        df = pd.read_csv(self.path, parse_dates=[[2,3]], index_col=0)
        df = df.iloc[:,2:] #drop useless  columns !set-10000 for testing purposes
        df = df.rename(
            columns={'<OPEN>': 'Open',
                    '<HIGH>': 'High',
                    '<LOW>': 'Low',
                    '<CLOSE>': 'Close',
                    '<VOL>': 'Volume'})
        df.index = df.index.rename('Date')
        df = df.iloc[-500:,:4] #drop volume
        return df

    @property
    def data(self):
        self.__data = self.__data.dropna()
        self.__data = self.__data.round(self.accuracy)
        return self.__data

    def add_rolling_average(self, period: int = 10):
        """Add extra column (rolling average) to dataframe"""
        self.__data[f'Average{period}'] = self.__data['Close'].rolling(period).mean()
        self.plot_info['plot'].append(self.__data.columns[-1])    

    def add_standart_deviation(self, period: int = 10): 
        """Add extra column (rolling standart deviation) to dataframe"""    
        self.__data[f'Deviation{period}'] = self.__data['Close'].rolling(period).std()
        self.plot_info['subplot'].append(self.__data.columns[-1])

    def add_corelation(self, period: int = 10): 
        """Add extra column (rolling corelation) to dataframe"""    
        self.__data[f'Corelation{period}'] = self.__data['Close'].rolling(period).mean()/self.__data['Close'].rolling(period).std()
        self.plot_info['subplot'].append(self.__data.columns[-1])

    def plot(self, volume: bool = False, from_date: str = None, to_date: str = None):
        add = []
        for col in self.plot_info['plot']:
            add.append(mpf.make_addplot(self.data.loc[from_date:to_date, col]))
        for col in self.plot_info['subplot']:
            add.append(mpf.make_addplot(self.data.loc[from_date:to_date, col], panel=1))

        mpf.plot(self.data.loc[from_date:to_date,:], type='candle', volume=volume, warn_too_much_data=50000, addplot=(add))
    
        mpf.show()

    def white_space(self, word):
        longest_name = max(self.data.columns.to_list(), key=lambda x: len(x))
        num_spaces = len(longest_name) - len(word)
        return ' '*num_spaces

    def __repr__(self):
        info = '\n'.join([
            f'\nDATA INFO:',
            f'{type(self.data)}',
            f'DatetimeIndex: {self.data.index.argmax()} entries, \nfrom {self.data.index[0]} to {self.data.index[-1]}',
            f'Data columns (total {len(self.data.columns)}):',
            f'------------------------------------',
            f'# |  Name{self.white_space("Name")}  |  Type  ',
            f'------------------------------------',
        ])           
        for col in self.data.columns:
            info = info + f'\n{self.data.columns.to_list().index(col)} |  {self.data[col].name}{self.white_space(col)}  |  {self.data[col].dtype}'

        info = info + f'\n------------------------------------'

        return info

class WindowGenerator():
    def __init__(self, input_width: int, label_width:int, shift:int, warm_up:int,
                 train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                 label_columns: str = ['Close'], n_algorythm: str = 'STD',):
        # Normalization options
        self.norm_algorythm = n_algorythm.lower()
        
        # Store data
        self.train = train_df
        self.val = val_df
        self.test = test_df

        # Work out the label column
        if label_columns is not None:
            self.label_columns = label_columns

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.warm_up = warm_up

        self.total_window_size = warm_up + input_width + shift
               
        self.input_slice = slice(warm_up, warm_up+ input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, self.total_window_size)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])
    
    def process_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, ...]:
        window = self.__normalization(self.__add_local_extrema(data))
        input = window[self.input_slice]
        label = window[self.label_slice]
        if self.label_columns is not None:
            label = label[self.label_columns]
        input = input.to_numpy()
        label = label.to_numpy().squeeze(axis=(1,))

        return input, label
       
    def __add_local_extrema(self, data : pd.DataFrame) ->  pd.DataFrame:
        """Add extra column (local min and max categorical type) to dataframe"""
        ndarray = data.iloc[0 : self.warm_up + self.input_width, -1].to_numpy()
        df = data.copy()
        loc_max_indx = argrelextrema(ndarray, np.greater)[0]
        loc_min_indx = argrelextrema(ndarray, np.less)[0]
        df['Loc_max'] = np.NaN
        df['Loc_min'] = np.NaN
        for indx in range(0, self.warm_up + self.input_width):
            df.iloc[indx, -1] = 1 if indx in loc_min_indx else 0
            df.iloc[indx, -2] = 1 if indx in loc_max_indx else 0
        
        df.loc[:, 'Loc_max'] = df.Loc_max.astype('category')
        df.loc[:, 'Loc_min'] = df.Loc_min.astype('category')

        return df

    def __normalization(self, data : pd.DataFrame) -> pd.DataFrame:
        norm_data = data.copy() 
        for col in norm_data.columns:
            if norm_data[col].dtype in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8']: 
                if self.norm_algorythm in ['std']:       
                    norm_data[col] = (data[col] - data[col].mean())/data[col].std() 
                elif self.norm_algorythm in ['minmax']:            
                    norm_data[col] = (data[col] - data[col].min())/(data[col].max()-data[col].min())
                else:
                    sys.exit('No normalization algorythm found or given')
        norm_data = norm_data.round(6)

        return norm_data
        

    def get_dataset(self, data):
        inputs = []
        labels = []
        for window in data.rolling(self.total_window_size):
            if len(window) == self.total_window_size:
                input, label = self.process_data(window)
                inputs.append(input)
                labels.append(label)
        
        return inputs, labels

    @property
    def train(self):
        return self.get_dataset(self.train)
    
    @property
    def val(self):
        return self.get_dataset(self.val)
    
    @property
    def test(self):
        return self.get_dataset(self.test)

    
