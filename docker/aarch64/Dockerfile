# docker build . --tag eschnett/nsimd-aarch64
# docker push eschnett/nsimd-aarch64
# docker run eschnett/nsimd-aarch64

FROM ubuntu:19.10

ENV DEBIAN_FRONTEND=noninteractive 

# John Milton was a polyglot
RUN mkdir /milton
WORKDIR /milton

# Install system packages
RUN apt-get update && \
    apt-get --yes --no-install-recommends install \
        apt-utils \
        build-essential \
        ca-certificates \
        clang-format \
        cmake \
        crossbuild-essential-arm64 \
        git \
        perl \
        python3 \
        qemu-user \
        && \
    rm -rf /var/lib/apt/lists/*

# TODO: install mpfr from scratch

COPY run.sh /milton/run.sh

CMD ./run.sh
