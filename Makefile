.DEFAULT_GOAL := help

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  reinstall_package  to install/reinstall the package"
	@echo "  run_django_server  to run the Django server"
	@echo "  run_fastapi        to run the FastAPI server"
	@echo "  run_django_tests   to run the Django tests"

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y mpdatanba || :
	@pip install -e .

run_fastapi:
	uvicorn mpdatanba.api.fast_api:app --reload

#################### SERVER ACTIONS ###################
run_django_server:
	@python backend/server/manage.py runserver

run django_tests:
	@python backend/server/manage.py test
