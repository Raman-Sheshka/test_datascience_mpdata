"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

# ML registry
import inspect
from ml.registry import MLRegistry
from mpdatanba.ml_logic.ml_workflow import MlModelWorkflow

try:
    registry = MLRegistry()  # create ML registry
    # Random Forest classifier
    ml_model_object = MlModelWorkflow()
    ml_model_object.load_model()
    # add to ML registry
    registry.add_algorithm(endpoint_name="income_classifier",
                           algorithm_object=ml_model_object,
                           algorithm_name="LGBMClassifier",
                           algorithm_status="production",
                           algorithm_version="0.0.1",
                           owner="Roman",
                           algorithm_description="LGBMClassifier with simple pre-processing",
                           algorithm_code=inspect.getsource(MlModelWorkflow)
                           )
except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))
