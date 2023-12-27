FROM ubuntu:latest
LABEL authors="frasr"

ENTRYPOINT ["top", "-b"]