import os
import re
import unidecode
import pandas as pd
import numpy as np
from typing import Union, List
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

def ts_slugify(ts : Union[str, pd.Series]):
    # slugify a pandas series or a string
    # remove special characters, accents and spaces
    # replace spaces by underscores
    # convert to lower case
    # it's a bit slow
    # TODO : use np.vectorize to make it faster
    if isinstance(ts, pd.Series):
        ts = ts.str.lower().str.replace(' ', '_')
        ts = ts.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        ts = ts.str.replace(pat="[^0-9A-z%_-]+", repl='', regex=True)
        ts = ts.str.replace(pat="['\\\\']", repl='', regex=True)
        return ts.str.strip('_')
    if isinstance(ts, str):
        string = ts.lower().replace(' ', '_')
        #replace % by percent
        string = string.replace('%', '_pca')
        string = unidecode.unidecode(string)
        string = re.sub('[^0-9A-z%_-]+', '', string)
        string = re.sub("['\\\\']", '', string)
        return string.strip('_')

from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score
from sklearn.model_selection import KFold


def score_classifier(dataset,
                     classifier,
                     labels,
                     nb_splits:int=3
                     ):

    """
    performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
    :param dataset: the dataset to work on
    :param classifier: the classifier to use
    :param labels: the labels used for training and validation
    :return:
    """

    kf = KFold(n_splits=nb_splits,
               random_state=50,
               shuffle=True
               )
    confusion_mat = np.zeros((2,2))
    recall = 0
    for training_ids,test_ids in kf.split(dataset):
        training_set = dataset[training_ids]
        training_labels = labels[training_ids]
        test_set = dataset[test_ids]
        test_labels = labels[test_ids]
        classifier.fit(training_set,training_labels)
        predicted_labels = classifier.predict(test_set)
        confusion_mat+=confusion_matrix(test_labels,predicted_labels)
        recall += recall_score(test_labels, predicted_labels)
    recall/=nb_splits

    return confusion_mat, recall
