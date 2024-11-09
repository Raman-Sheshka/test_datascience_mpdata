import pandas as pd
import numpy as np
import os
import joblib
from rich import print as rprint
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
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
        rprint(f"Data loaded from {self.__data_dir_path}")
        #i don't like the upper case in the column names, let's fix that
        df_raw = self.slugify_columns(df_raw)
        return df_raw

    @staticmethod
    def slugify_columns(df):
        raw_columns_list = df.columns.tolist()
        slug_columns_list = [ts_slugify(s) for s in raw_columns_list]
        rprint(f"columns : {raw_columns_list}")
        rprint("have been slugified to:")
        rprint(f"columns : { slug_columns_list}")
        #rename columns
        encoded_columns = [s_.replace('3', 'three_') for s_ in slug_columns_list]
        rprint(encoded_columns)
        df.columns = encoded_columns
        return df

class Preprocessor:
    SCALER_PATH_FILE = os.path.join(PARENT_BASE_PATH,
                                    'save_models',
                                    'scaler.pkl')
    def __init__(self,
                 features : List[str] = [],
                 target : str = 'target_5yrs',
                 id_col : str = 'name',
                 ):
        self.__features = features
        self.__target = target
        self.__id_col = id_col
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self._names_train = None
        self._names_test = None
        self._scaler = MinMaxScaler()



    def get_train_test_datasets(self, df:pd.DataFrame):
        x, y, names = self._build_dataset(df)
        self.build_test_dataset(x, y, names)


    def _build_dataset(self, df : pd.DataFrame):
        """
        Build a dataset from a dataframe
        :return: X, y, names
        """
        df = self.clean_data(df)
        assert self.__id_col in df.columns, f"{self.__id_col} not in columns"
        assert self.__target in df.columns, f"{self.__target} not in columns"

        # extract names, labels, features names and values
        names = df[self.__id_col].values.tolist() # players names
        y = df[self.__target].values # labels

        if not self.__features:
            self.__features = df.drop([self.__target,self.__id_col],axis=1).columns.values
            x = df.drop([self.__target,self.__id_col],axis=1).values
        else:
            assert all([f_ in df.columns for f_ in self.__features]), "features not in columns"
            assert self.__target not in self.__features, f"{self.__target} in features"
            assert self.__id_col not in self.__features, f"{self.__id_col} in features"
            x = df[self.__features].values

        return x, y, names

    def build_test_dataset(self,x, y, names):
        X_train, X_test, self._y_train, self._y_test,\
            self._names_train, self._names_test = train_test_split(x,y,names,
                                                                   test_size=0.1,
                                                                   random_state=42
                                                                    )
        # normalize the data
        self._X_train = self.build_normalized_set(X_train)
        self._X_test = self._scaler.transform(X_test)
        return

    def build_normalized_set(self,data):
        # scaler is trained
        norm_data = self._scaler.fit_transform(data)
        self.save_scaler()
        return norm_data


    def save_scaler(self):
        # save scaler path
        check_is_fitted(self._scaler)
        joblib.dump(self._scaler, self.SCALER_PATH_FILE)
        rprint(f"Scaler saved in {self.SCALER_PATH_FILE}")
        return

    def load_scaler(self):
        # load the scaler
        try:
            scaler = joblib.load(self.SCALER_PATH_FILE)
            return scaler
        except Exception as e:
            rprint(f"Error loading scaler {e}")
            return None


    @staticmethod
    def clean_data(data_:pd.DataFrame) -> pd.DataFrame:

        data_df = data_.copy(deep=True)
        #replace all nan values with 0.0
        column_name = 'three_p_pca'
        array_clean = np.nan_to_num(data_df[column_name].values,nan=0.0)
        data_df[column_name] = array_clean
        assert all(data_df.isna().sum().values == 0)

        #drop duplicates
        data_df.drop_duplicates(inplace=True)
        data_df.reset_index(drop=True, inplace=True)
        data_df.drop_duplicates(subset=['name'], keep='last', inplace=True)
        data_df.reset_index(drop=True, inplace=True)
        assert data_df[data_df.duplicated(keep=False)].sort_values(by='name').shape[0] == 0, "There are still duplicated rows"
        return data_df

    def __str__(self):
        return f"Preprocessor: id {self.__id_col}, target {self.__target}"

    def __repr__(self):
        return f"Preprocessor: id {self.__id_col}, target {self.__target}"
