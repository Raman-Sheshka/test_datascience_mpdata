import glob
import os
import time
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from mpdatanba.ml_logic.preprocessor import Loader, Preprocessor
from mpdatanba.utils import create_folder_if_not_exists
from mpdatanba import PARENT_BASE_PATH
import gc

class MLModelWorkflow:
    # the class serve to train, evaluate, save and load the model #
    # the model is saved in the save_models directory #
    # the encoder was saved in the save_models directory during the preprocessing #
    # the model and encored are loaded from the save_models directory #
    def __init__(self,
                 model_type:str='sklearn', # should be declared in config
                 mode:str='debug', # should be declared in config
                 ):
        self.model_type = model_type
        self.execution_mode = mode
        self.threshold = 0.5
        self.model_dir_path = self.get_model_dir_path()
        self.model = None

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train_model(self,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    ):
        """
        Train the model
        :param X : numpy array, input data, preprocessed
        :param y : numpy array, target data
        """
        # Train the model
        print("Training model...")
        score_confusion_matrix, score_recall = self.score_classifier(X_train,
                                                                   y_train
                                                                   )
        self.model.fit(X_train, y_train)
        recall_, precision_, accuracy_ = self.evaluate_model(X_test,
                                                             y_test
                                                           )
        print("Model trained!")
        return score_confusion_matrix, score_recall, recall_, precision_, accuracy_

    def score_classifier(self,
                        X,
                        y,
                        nb_splits:int=3
                        ):

        """
        performs 3 random trainings/tests to build a confusion matrix and prints results with precision and recall scores
        :param dataset: the dataset to work on
        :param labels: the labels used for training and validation
        :return:
        """
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert isinstance(y, np.ndarray), "y must be a numpy array"
        assert self.model is not None, "Model is not loaded"

        if self.model_type == 'sklearn':
            kf = KFold(n_splits=nb_splits,
                    random_state=50,
                    shuffle=True
                    )
            confusion_mat = np.zeros((2,2))
            recall = 0
            for training_ids,test_ids in kf.split(X):
                training_set = X[training_ids]
                training_labels = y[training_ids]
                test_set = X[test_ids]
                test_labels = y[test_ids]
                self.model.fit(training_set,training_labels)
                predicted_labels = self.model.predict(test_set)
                confusion_mat+=confusion_matrix(test_labels,predicted_labels)
                recall += recall_score(test_labels, predicted_labels)
            recall/=nb_splits

        return confusion_mat, recall

    def evaluate_model(self,
                        X_test,
                        y_test,
                        ):
        """
        Evaluate the model
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: confusion matrix, recall, precision, accuracy
        """
        y_pred = self.get_predict(X_test)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        return recall, precision, accuracy

    def get_predict(self,
                    X,
                    ) :
        print("Making predictions...")
        assert self.model is not None, "Model is not loaded"
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        y_pred = self.model.predict(X)

        if self.model_type == 'lgbm':
            y_pred = (y_pred > self.threshold).astype(int)
        print("Predictions made!")
        return y_pred

    def load_model(self):
        """Load the latest model from disk
        Args:
            model_target (str, optional): config params. Defaults to 'local'.
            sklearn (bool, optional): config params. Defaults to False.

        Returns:
            : latest saved model
        """
        # Load the model
        print("Loading model...")
        # Get the latest model version name by the timestamp on disk
        local_model_paths = glob.glob(f"{self.model_dir_path}/*")

        assert local_model_paths, "No model found on local disk"

        most_recent_model_path_on_disk = sorted(local_model_paths)[0]
        if self.model_type == 'lgbm':
            latest_model = lgb.Booster(model_file=most_recent_model_path_on_disk)
        if self.model_type == 'sklearn':
            latest_model = joblib.load(most_recent_model_path_on_disk)
        print("Model loaded from local disk")
        self.model = latest_model
        return

    def save_model(self):
        # Save the model
        print("Saving model...")
        assert self.model is not None, "Load Model or Train Model before saving"
        create_folder_if_not_exists(self.model_dir_path)
        if self.model_type == 'lgbm':
            model_file = os.path.join(self.model_dir_path,
                                "model_selected.txt"
                                )
            self.model.save_model(model_file)
        if self.model_type == 'sklearn':
            model_file = os.path.join(self.model_dir_path,
                                "model_selected.pkl"
                                )
            joblib.dump(self.model, model_file)
        print("Model saved!")
        return

    def get_model_dir_path(self):
        if self.execution_mode == "debug":
           return os.path.join(PARENT_BASE_PATH,"save_models")

class MLWorkflow(MLModelWorkflow):
    def __init__(self,
                 model_type:str='sklearn',
                 mode:str='debug',
                 ):
        super(MLWorkflow, self).__init__(model_type, mode)
        self.encoder = None
        self._X_train = None
        self._y_train = None
        self._X_test = None
        self._y_test = None
        self.metrics = {}

    def set_encoder(self, encoder):
        self.encoder = encoder

    def build_dataset(self):
        loader = Loader()
        preprocessor = Preprocessor()
        preprocessor.get_train_test_datasets(loader._df_raw)
        self._X_train = preprocessor._X_train
        self._y_train = preprocessor._y_train
        self._X_test = preprocessor._X_test
        self._y_test = preprocessor._y_test
        self.encoder = preprocessor._scaler
        del loader, preprocessor

    def preprocess_data(self, x):
        # Preprocess the data
        print("Preprocessing data...")
        res = self.encoder.transform(x)
        print("Data preprocessed!")
        return res

    def load_encoder(self):
        # Load the encoder
        print("Loading encoder...")
        encoder_file = os.path.join(self.model_dir_path,
                                "scaler.pkl"
                                )
        try:
            scaler = joblib.load(encoder_file)
            print("Encoder loaded!")
            return scaler
        except FileNotFoundError:
            print("No encoder found on local disk")
            return

    def full_train_model(self,
                         model,
                         partial_train:bool=False,
                         save_model:bool=False,

                         ):
        """
        Train the model
        :param model : sklearn model to train
        :param save_model : bool, save the model or not
        """
        # Train the model
        self.set_model(model)
        print("Full training model...")
        if partial_train:
            assert self._X_train is not None, "No dataset found, please build the dataset first"
            assert self._y_train is not None, "No dataset found, please build the dataset first"
            assert self._X_test is not None, "No dataset found, please build the dataset first"
            assert self._y_test is not None, "No dataset found, please build the dataset first"
        else:
            self.build_dataset()
        score_confusion_matrix,score_recall, recall_, precision_, accuracy_ = self.train_model(self._X_train,
                                                                                               self._y_train,
                                                                                               self._X_test,
                                                                                               self._y_test
                                                                                               )
        self.metrics["score_confusion_matrix"] = score_confusion_matrix
        self.metrics["score_recall"] = score_recall
        self.metrics["recall"] = recall_
        self.metrics["precision"] = precision_
        self.metrics["accuracy"] = accuracy_
        if save_model:
            self.save_model()
        gc.collect()
        print("Full model trained!")
        return

    def compute_predict(self,
                    X,
                    ) :
        # model inference
        print("Making predictions...")
        if self.encoder is None:
            self.encoder = self.load_encoder()
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = self.preprocess_data(X)
        y_pred = self.get_predict(X)
        return y_pred
