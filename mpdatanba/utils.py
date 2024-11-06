import os
import pandas as pd
from typing import List
from itertools import compress

def check_if_list_of_columns_exist(df: pd.DataFrame,
                                   columns_list: List[str]
                                   ) -> bool:
    return all([c_ in df.columns for c_ in columns_list])


def df_drop_columns(df:pd.DataFrame,
                    columns_to_drop:List[str]
                    ) -> pd.DataFrame:
    df_ = df.copy()
    mask = [col in df_.columns for col in columns_to_drop]
    return df_.drop(columns = list(compress(columns_to_drop, mask)))

def create_folder_if_not_exists(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"Folder {folder_name} already exists")
