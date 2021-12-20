# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.209.6/containers/python-3-miniconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yaml requirements.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yaml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yaml; fi \
    && rm -rf /tmp/conda-tmp

RUN useradd -rm -d /home/user -s /bin/bash -g root -G sudo -u 1001 user
USER user
RUN mkdir /home/user/transductions
WORKDIR /home/user/transductions 
