import numpy as np
import inspect
from django.test import TestCase
from ml.classifier import Classifier
from ml.registry import MLRegistry

class MLDjangoTests(TestCase):

    def test_model_instantiation(self):
        my_alg = Classifier()
        self.assertIsNotNone(my_alg.model)
        self.assertIsNotNone(my_alg.encoder)

    def test_ml_algorithm(self):
        test_data = {
                    "gp": 77.0,
                    "min": 14.3,
                    "pts": 5.2,
                    "fgm": 1.9,
                    "fga": 4.0,
                    "fg_pca": 47.6,
                    "three_p_made": 0.0,
                    "three_pa": 0.0,
                    "three_p_pca": 0.0,
                    "ftm": 1.4,
                    "fta": 2.2,
                    "ft_pca": 63.2,
                    "oreb": 1.1,
                    "dreb": 2.4,
                    "reb": 3.5,
                    "ast": 0.5,
                    "stl": 0.3,
                    "blk": 0.4,
                    "tov": 1.1
                }
        my_alg = Classifier()
        response = my_alg.compute_predict(test_data)
        self.assertEqual(response['status'], 'OK')
        self.assertTrue('label' in response)
        self.assertEqual(response['label'], 'Yes')
        #self.assertIsNotNone(prediction)
        #self.assertIn(prediction, [1.0])

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)
        endpoint_name = "income_classifier"
        algorithm_object = Classifier()
        algorithm_name = "LGBMClassifier"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "Roman"
        algorithm_description = "LGBMClassifier with simple pre-processing"
        algorithm_code = inspect.getsource(Classifier)
        # add to registry
        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                    algorithm_status, algorithm_version, algorithm_owner,
                    algorithm_description, algorithm_code)
        # there should be one endpoint available
        self.assertEqual(len(registry.endpoints), 1)
