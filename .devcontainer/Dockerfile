FROM mcr.microsoft.com/devcontainers/anaconda:1-3

COPY .devcontainer/environment.yml /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then umask 0002 && /opt/conda/bin/conda env update -n base -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp

RUN git config --global core.autocrlf true
RUN git config --global user.email "mazza.tommaso@gmail.com"
RUN git config --global user.name "tm4zza"

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
