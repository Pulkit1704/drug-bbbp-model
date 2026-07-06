# Makefile
.PHONY: lock build

lock:
	conda-lock --mamba -f environment.prod.yml -p linux-64

build: 
	$(DOCKER) build -f dockerfile -t gnn-fastapi-app .