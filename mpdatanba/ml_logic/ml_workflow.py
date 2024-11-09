import glob
import os
import time
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.metrics import recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from mpdatanba.utils import create_folder_if_not_exists
from mpdatanba import PARENT_BASE_PATH


class MlModelWorkflow:
    # the class serve to train, evaluate, save and load the model #
    def __init__(self,
                 model_type:str='sklearn', # should be declared in config
                 mode:str='debug', # should be declared in config
                 ):
        self.model_type = model_type
        self.execution_mode = mode
        self.__threshold = 0.5
        self.__model_dir_path = self.get_model_dir_path()
        self.model = None
        self.encoder = self.load_encoder()

    def set_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def train_model(self):
        # Train the model
        print("Training model...")
        time.sleep(2)
        print("Model trained!")
        return None

    def preprocess_data(self, x):
        # Preprocess the data
        print("Preprocessing data...")
        return self.encoder.transform(x)

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
        y_pred = self.compute_predict(X_test)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        return recall, precision, accuracy

    def compute_predict(self,
                    X,
                    ) :
        print("Making predictions...")
        assert self.model is not None, "Model is not loaded"
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        X = self.preprocess_data(X)
        y_pred = self.model.predict(X)

        if self.model_type == 'lgbm':
            y_pred = (y_pred > self.__threshold).astype(int)
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
        local_model_paths = glob.glob(f"{self.__model_dir_path}/*")

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
        create_folder_if_not_exists(self.__model_dir_path)
        if self.model_type == 'lgbm':
            model_file = os.path.join(self.__model_dir_path,
                                "model_selected.txt"
                                )
            self.model.save_model(model_file)
        if self.model_type == 'sklearn':
            model_file = os.path.join(self.__model_dir_path,
                                "model_selected.pkl"
                                )
            joblib.dump(self.model, model_file)
        print("Model saved!")
        return

    def load_encoder(self):
        # Load the encoder
        print("Loading encoder...")
        encoder_file = os.path.join(self.__model_dir_path,
                                "scaler.pkl"
                                )
        try:
            scaler = joblib.load(encoder_file)
            print("Encoder loaded!")
            return scaler
        except FileNotFoundError:
            print("No encoder found on local disk")
            return

    def get_model_dir_path(self):
        if self.execution_mode == "debug":
           return os.path.join(PARENT_BASE_PATH,"save_models")
