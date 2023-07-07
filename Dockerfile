FROM python:3.10-slim
EXPOSE 8080
RUN apt-get update && apt-get -y install build-essential
ADD ./requirements.lock /
RUN pip install -r /requirements.lock
ARG GATEWAY
ENV GATEWAY=$GATEWAY
ADD . /plugin
ENV PYTHONPATH=$PYTHONPATH:/plugin
WORKDIR /plugin/services
CMD python services.py