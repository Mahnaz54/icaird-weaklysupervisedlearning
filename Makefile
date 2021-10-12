#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = icaird-weaklysupervisedlearning
PYTHON_INTERPRETER = python
PYTHON_VERSION = 3.6

## test if Anaconda is installed
ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

# network
JUPYTER_PORT := 8361
#################################################################################
# PYTHON ENVIRONMENT COMMANDS                                                   #
#################################################################################

## set up the python environment
create_environment:
ifeq (True,$(HAS_CONDA))
	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION)
	@echo ">>> New conda env created. Activate with:\nsource activate $(PROJECT_NAME)"
else
	@echo "Conda is not installed. Please install it."
endif

## install the requirements into the python environment
requirements: install_curl   
	conda env update --file environment.yml
	pip install -r requirements.txt

## save the python environment so it can be recreated
export_environment:
	conda env export --no-builds | grep -v "^prefix: " > environment.yml
	# note - the requirements.txt. is required to build the
	# environment up but is not changed are part of the export
	# process

# some packages that are required by the project have binary dependencies that
# have to be installed out with Conda.
install_isyntax_sdk:
	sudo apt install gdebi -y
	sudo gdebi -n ./libraries/philips-pathology-sdk/*pixelengine*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*eglrendercontext*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*gles2renderbackend*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*gles3renderbackend*.deb
	sudo gdebi -n ./libraries/philips-pathology-sdk/*softwarerenderer*.deb
		
install_curl:
	sudo apt -y install curl

install_git:
	sudo apt-get install git

install_topk:
	git clone https://github.com/oval-group/smooth-topk.git
	cd smooth-topk
	python setup.py install
	cd ..

install_docker:
	# this installs Docker Community Edition from the official Docker repository
	sudo apt update
	sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
	sudo apt-get update
	sudo apt-get -y install docker-ce
	# then we are going to add the current user to the Docker group so we can connect to the docker
	# process when we are not root
	sudo groupadd docker
	sudo usermod -aG docker $USER
	echo "Please logout and log back in for changes to take effect :D"

docker_image:
	docker build -t $(PROJECT_NAME) .

docker_run:
	docker run --shm-size=8G --gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/ubuntu/$(PROJECT_NAME) \
				-v /data3/iCAIRD:/home/ubuntu/$(PROJECT_NAME)/data \
				-v /data1/icaird-weaklysupervisedlearning/results:/home/ubuntu/$(PROJECT_NAME)/results \
				-v /data1/icaird-weaklysupervisedlearning/DATA_ROOT_DIR:/home/ubuntu/$(PROJECT_NAME)/DATA_ROOT_DIR \
				-it $(PROJECT_NAME):latest

docker_run_local:
	docker run --shm-size=16G --gpus all -p $(JUPYTER_PORT):$(JUPYTER_PORT) \
				-v $(PROJECT_DIR):/home/ubuntu/$(PROJECT_NAME) \
				-it $(PROJECT_NAME):latest
	

#################################################################################
# JUPYTER COMMANDS                                                              #
#################################################################################
setup_jupyter:
	pip install --user ipykernel
	python -m ipykernel install --user --name=$(PROJECT_NAME)

run_notebook:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root

run_lab_tb:
	jupyter lab --ip=* --port $(JUPYTER_PORT) --allow-root


