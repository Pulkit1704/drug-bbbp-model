
# Builder Environment

FROM mambaorg/micromamba:1.5-bullseye-slim AS builder

USER root
WORKDIR /build

COPY conda-lock.yml .

RUN micromamba install -y --name base --file conda-lock.yml 

ENV PATH="/opt/conda/bin:$PATH"

RUN micromamba clean --all --yes && \
    find /opt/conda -type d -name "__pycache__" -exec rm -rf {} +


# Running enviornment 
FROM mambaorg/micromamba:1.5-bullseye-slim AS runner

USER root
WORKDIR /app

COPY --from=builder /opt/conda /opt/conda

COPY --chown=$MAMBA_USER:$MAMBA_USER ./app /app

EXPOSE 7860

ENV PATH="/opt/conda/bin:$PATH"

USER $MAMBA_USER

CMD ["python", "main.py"]