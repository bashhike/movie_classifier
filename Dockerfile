FROM python:3.8-slim-buster
MAINTAINER Abhishek Pandey <ashp.pandey916@gmail.com>

ENV HOME=/home/user

COPY . ${HOME}/movie_classifier
WORKDIR ${HOME}/movie_classifier

# Install required packages.
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm

# Add a default user for running the container.
RUN useradd -u 1001 -r -g 0 -md ${HOME} -s /sbin/nologin -c "Default user" default \
    && chgrp -R 0 ${HOME} \
    && chown -R 1001 ${HOME} \
    && chmod -R 755 ${HOME}

# Run the container as a non-root user to avoid security issues.
USER 1001

ENTRYPOINT ["bash"]