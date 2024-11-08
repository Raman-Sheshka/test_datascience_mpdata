.DEFAULT_GOAL := default

#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y mpdatanba || :
	@pip install -e .

default:
	@echo 'Please specify a target to run'

# run_restapi:
#   @python -c 'from mpdatanba.api.rest_api.py

run_fastapi:
	uvicorn mpdatanba.api.fast_api:app --reload
