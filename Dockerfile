FROM ubuntu:noble

RUN apt update && apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
RUN apt update && apt install -y build-essential pkg-config libcairo2-dev libgirepository-2.0-dev

SHELL ["/bin/bash", "--login", "-i", "-c"]

USER ubuntu

RUN curl -fsSL https://pyenv.run | bash
ENV PATH="~/.pyenv/bin:$PATH"
RUN pyenv init 2>> ~/.bashrc || echo done

RUN mkdir /home/ubuntu/workspace
WORKDIR /home/ubuntu/workspace

RUN pyenv install 3.9.21
RUN pyenv local 3.9.21

# this shouldn't be required because numpy is in requirements.txt, but it is.
# Some depended on a package somewhere requires it for its setup script and fails
RUN pip install numpy 
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY ./chatterboxGenServer.py ./chatterboxGenServer.py
COPY ./voiceToy.py ./voiceToy.py