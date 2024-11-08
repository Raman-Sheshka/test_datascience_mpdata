import numpy as np
import inspect
from django.test import TestCase
from mpdatanba.ml_logic.ml_workflow import MlModelWorkflow
from ml.registry import MLRegistry

# Create your tests here.
class MLDjangoTests(TestCase):
    def test_ml_algorithm(self):
        test_data = {
                    'gp': 0.9295774647887325,
                    'min': 0.29629629629629634,
                    'pts': 0.16363636363636364,
                    'fgm': 0.16161616161616163,
                    'fga': 0.16842105263157894,
                    'fg_pca': 0.47695390781563124,
                    'three_p_made': 0.0,
                    'three_pa': 0.0,
                    'three_p_pca': 0.0,
                    'ftm': 0.1818181818181818,
                    'fta': 0.21568627450980396,
                    'ft_pca': 0.632,
                    'oreb': 0.2075471698113208,
                    'dreb': 0.23404255319148934,
                    'reb': 0.23529411764705885,
                    'ast': 0.04716981132075472,
                    'stl': 0.12,
                    'blk': 0.10256410256410259,
                    'tov': 0.23255813953488375
                    }
        my_alg = MlModelWorkflow()
        my_alg.load_model() # load the latest model
        prediction = my_alg.compute_predict(np.array([list(test_data.values())]))
        self.assertIsNotNone(prediction)
        self.assertIn(prediction, [1.0])


    def test_model_instantiation(self):
        model = MlModelWorkflow()
        model.load_model()
        self.assertIsNotNone(model.model)

    # def test_model_train(self):
    #     model = MlModelWorkflow()
    #     model.train_model()
    #     self.assertIsNotNone(model.get_model())

    # def test_model_score(self):
    #     model = MlModelWorkflow()
    #     model.train_model()
    #     model.score_classifier(X=[[1, 2], [3, 4]], y=[0, 1])
    #     self.assertIsNotNone(model.get_model())

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = MlModelWorkflow()
        algorithm_name = "LGBMClassifier"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Roman"
        algorithm_description = "LGBMClassifier with simple pre-processing"
        algorithm_code = inspect.getsource(MlModelWorkflow)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
