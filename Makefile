.DEFAULT_GOAL := help

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  reinstall_package  to install/reinstall the package"
	@echo "  run_fastapi        to run the FastAPI client"

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y mpdatanba || :
	@pip install -e .

run_fastapi:
	uvicorn mpdatanba.api.fast_api:app --reload

run_flask:
	@python flask/app.py
