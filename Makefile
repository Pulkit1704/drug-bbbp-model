# Makefile
.PHONY: lock build

lock:
	conda-lock --mamba -f environment.prod.yml -p linux-64

build: 
	$(DOCKER) build -f Dockerfile -t gnn-fastapi-app .