FROM continuumio/miniconda3:latest

WORKDIR /app

# Install mamba
RUN conda install -y mamba -n base -c conda-forge

# Copy environment file
COPY services/shap-service/environment.yml /app/environment.yml

# Create conda environment
RUN mamba env create -f /app/environment.yml && \
    mamba clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "shap-service", "/bin/bash", "-c"]

# Copy application code
COPY services/shap-service/ /app/
COPY shared/ /app/shared/

# Set environment variables
ENV PATH /opt/conda/envs/shap-service/bin:$PATH
ENV PYTHONPATH /app:$PYTHONPATH

# Run the application
CMD ["conda", "run", "-n", "shap-service", "python", "app/main.py"]


