FROM continuumio/miniconda3:latest

WORKDIR /app

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/training-service/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "training-service", "/bin/bash", "-c"]

# Copy application code
COPY services/training-service/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/training-service/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["conda", "run", "--no-capture-output", "-n", "training-service", "python", "-u", "app/main.py"]

