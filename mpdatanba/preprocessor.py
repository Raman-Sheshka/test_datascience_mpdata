import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from mpdatanba import PARENT_BASE_PATH
from mpdatanba.utils import ts_slugify

class Loader():
    """it's a simple class to load the data, the data are stored in the data folder data
    the data sre read with pandas and return a pandas dataframe
    NOTE : the data are not preprocessed.
           it's possible use a class call method
    """
    def __init__(self):
        self.__data_dir_path = os.path.join(PARENT_BASE_PATH, 'data')
        self.__data_file_name = 'nba_logreg.csv'
        self._df_raw = self.load_data()

    def load_data(self):
        assert os.path.exists(self.__data_dir_path), f"Data directory {self.__data_dir_path} does not exist"
        assert os.path.exists(os.path.join(self.__data_dir_path, self.__data_file_name)), f"Data file {self.__data_file_name} does not exist"
        df_raw = pd.read_csv(os.path.join(self.__data_dir_path,
                                        self.__data_file_name
                                        )
                           )
        print(f"Data loaded from {self.__data_dir_path}")
        #i don't like the upper case in the column names, let's fix that
        df_raw = self.slugify_columns(df_raw)
        return df_raw

    @staticmethod
    def slugify_columns(df):
        raw_columns_list = df.columns.tolist()
        slug_columns_list = [ts_slugify(s) for s in raw_columns_list]
        print(f"columns : {raw_columns_list}")
        print("have been slugified to:")
        print(f"columns : { slug_columns_list}")
        #rename columns
        df.columns = slug_columns_list
        return df

from typing import List

class Preprocessor:
    def __init__(self,
                 df : pd.DataFrame,
                 features : List[str] = [],
                 target : str = 'target_5yrs',
                 id_col : str = 'name',
                 ):
        self.__df_raw = df
        self.__features = features
        self.__target = target
        self.__id_col = id_col
        self._df = None
        self._X = None
        self._y = None
        self._names = None
        self.__normalize = True
        self._scaler = MinMaxScaler()


    def build_dataset(self):
        """
        Build a dataset from a dataframe
        :return: X, y, names
        """
        assert self.__df_raw is not None, "df is None"
        assert self.__id_col in self.__df_raw.columns, f"{self.__id_col} not in columns"
        assert self.__target in self.__df_raw.columns, f"{self.__target} not in columns"

        # extract names, labels, features names and values
        self._names = self.__df_raw[self.__id_col].values.tolist() # players names
        self._y = self.__df_raw[self.__target].values # labels

        if not self.__features:
            self.__features = self.__df_raw.drop([self.__target,self.__id_col],axis=1).columns.values
            df_vals = self.__df_raw.drop([self.__target,self.__id_col],axis=1).values
        else:
            assert all([f_ in self.__df_raw.columns for f_ in self.__features]), "features not in columns"
            assert self.__target not in self.__features, f"{self.__target} in features"
            assert self.__id_col not in self.__features, f"{self.__id_col} in features"
            df_vals = self.__df_raw[self.__features].values

        self._X = df_vals
        return self._X, self._y, self._names

    def build_test_dataset(self):
        assert self._X is not None, "X is None"
        X_train, self._X_test, y_train, self._y_test = train_test_split(self._X,
                                                                        self._y,
                                                                        test_size=0.1,
                                                                        random_state=42
                                                                        )
        # normalize the data
        if self.__normalize:
            X_train = self.build_normalized_set(X_train)
            self._X_test = self._scaler.transform(self._X_test)
        return X_train, self._X_test, y_train, self._y_test

    def build_normalized_set(self,
                             data
                             ):
        # scaler is trained
        norm_data = self._scaler.fit_transform(data)
        return norm_data


    def save_scaler(self):
        # save the scaler
        pass

    def load_scaler(self):
        # load the scaler
        pass


    def __str__(self):
        return f"Preprocessor with {len(self.features)} features"

    def __repr__(self):
        return f"Preprocessor with {len(self.features)} features"
