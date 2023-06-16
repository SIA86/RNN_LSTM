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
        norm_algorythm : str
            set algorythm for data normalization (default: None, available: 'std', 'minmax')   

        Returns:
        ----------
            data - pd.DataFrame raw data

        """                            
        self.path = path
        self.accuracy = accuracy
        self.norm_algorythm = norm_algorythm

        self.__data = self.__get_data()
        self.std = self.__data.std()
        self.mean = self.__data.mean()
        self.min = self.__data.min()
        self.max = self.__data.max()
        

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
        df = df.iloc[:,:4] #drop volume
        return df

    @property
    def data(self):
        self.__data = self.__data.dropna()
        self.__data = self.__data.round(self.accuracy)
        return self.__data

    def add_rolling_average(self, period: int = 10):
        """Add extra column (rolling average) to dataframe"""
        self.__data[f'Average{period}'] = self.__data['Close'].rolling(period).mean()   

    def add_standart_deviation(self, period: int = 10): 
        """Add extra column (rolling standart deviation) to dataframe"""    
        self.__data[f'Deviation{period}'] = self.__data['Close'].rolling(period).std()

    def add_corelation(self, period: int = 10): 
        """Add extra column (rolling corelation) to dataframe"""    
        self.__data[f'Corelation{period}'] = self.__data['Close'].rolling(period).mean()/self.__data['Close'].rolling(period).std()

    def add_local_extrema(self):
        """Add extra column (local min and max categorical type) to dataframe"""
        ndarray = self.__data['Close'].to_numpy()
        loc_max_indx = argrelextrema(ndarray, np.greater)
        loc_min_indx = argrelextrema(ndarray, np.less)
        self.__data['Loc_max'] = 0
        for el in loc_max_indx:
            self.__data.iloc[el, -1] = 1
        self.__data['Loc_min'] = 0
        for el in loc_min_indx:
            self.__data.iloc[el, -1] = 1

        self.__data['Loc_max'] = self.__data.Loc_max.astype('category')
        self.__data['Loc_min'] = self.__data.Loc_min.astype('category')

    def normalization(self, data : pd.DataFrame) -> pd.DataFrame:
        if type(data).__name__ != 'NoneType':
            norm_data = data.copy() 
            for col in norm_data.columns:
                if norm_data[col].dtype in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8']: 
                    if self.norm_algorythm in ['std']:       
                        self.mean[col] = data[col].mean()
                        self.std[col] = data[col].std()
                        norm_data[col] = (data[col] - self.mean[col])/self.std[col] 
                    elif self.norm_algorythm in ['minmax']:            
                        self.max[col] = data[col].max()
                        self.min[col] = data[col].min()
                        norm_data[col] = (data[col] - self.min[col])/(self.max[col]-self.min[col])
                    else:
                        sys.exit('No normalization algorythm found or given')
            norm_data = norm_data.round(self.accuracy)
            return norm_data
        return 

    def reverse_norm(self, data: pd.Series) -> pd.Series:
        """Put predicted values and make reverse normalization  """   
        name = data.name.removeprefix('Pred_')
        if self.norm_algorythm in ['minmax']:
            rn_data = data * (self.max[name]-self.min[name]) + self.min[name]
        elif self.norm_algorythm in ['std']:
            rn_data = data * self.std[name] + self.mean[name]

        return rn_data

    def plot(self, volume: bool = False, from_date: str = None, to_date: str = None, extra_columns: list = []):
        if extra_columns:
            add = []
            for col in extra_columns:
                add.append(mpf.make_addplot(self.data.loc[from_date:to_date, col]))
            mpf.plot(self.data.loc[from_date:to_date,:], type='candle', volume=volume, warn_too_much_data=50000, addplot=(add))
        else:
            mpf.plot(self.data.loc[from_date:to_date,:], type='candle', volume=volume, warn_too_much_data=50000)

        mpf.show()

    def white_space(self, word):
        longest_name = max(self.data.columns.to_list(), key=lambda x: len(x))
        num_spaces = len(longest_name) - len(word)
        return ' '*num_spaces

    def __repr__(self):
        info = '\n'.join([
            f'\nDATA INFO:',
            f'{type(self.data)}',
            f'DatetimeIndex: {self.data.index.argmax()} entries, from {self.data.index[0]} to {self.data.index[-1]}',
            f'Normalization algorythm - {self.norm_algorythm if not self.norm_algorythm else self.norm_algorythm.upper()}',
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
    def __init__(self, input_width: int, label_width: int, shift: int,
                train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                label_columns: str = None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack([labels[:, :, self.column_indices[name]]
                               for name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels


    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [15 min]')
        plt.show()


    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result



    
