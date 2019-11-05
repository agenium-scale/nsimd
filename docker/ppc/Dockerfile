# docker build . --tag eschnett/nsimd-ppc
# docker push eschnett/nsimd-ppc
# docker run eschnett/nsimd-ppc

FROM ubuntu:19.10

ENV DEBIAN_FRONTEND=noninteractive 

# John Milton was a polyglot
RUN mkdir /milton
WORKDIR /milton

# Install system packages
RUN apt-get update && \
    apt-get --yes --no-install-recommends install \
        binutils-powerpc64-linux-gnu \
        build-essential \
        ca-certificates \
        clang-format \
        cmake \
        crossbuild-essential-ppc64el \
        git \
        libfdt-dev \
        libglib2.0-dev \
        libpixman-1-dev \
        perl \
        pkg-config \
        python \
        python3 \
        wget \
        zlib1g-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Install qemu

# We can't use Ubuntu's qemu, nor the released versions. This
# PPC-specific bug <https://bugs.launchpad.net/qemu/+bug/1841990>
# might be the reason.

# qemu-4.1.0: illegal instruction
# RUN mkdir src && \
#     ( cd src && \
#     wget https://download.qemu.org/qemu-4.1.0.tar.xz && \
#     tar xJf qemu-4.1.0.tar.xz \
#     cd qemu-4.1.0 && \
#     ./configure --target-list=ppc64le-linux-user --prefix=/milton/qemu && \
#     make -j$(nproc) && \
#     make -j$(nproc) install && \
#     true ) && \
#     rm -rf src

# qemu-4.0.1: build failure
# RUN mkdir src && \
#     ( cd src && \
#     wget https://download.qemu.org/qemu-4.0.1.tar.xz && \
#     tar xJf qemu-4.0.1.tar.xz && \
#     cd qemu-4.0.1 && \
#     ./configure --target-list=ppc64le-linux-user --prefix=/milton/qemu && \
#     make -j$(nproc) && \
#     make -j$(nproc) install && \
#     true ) && \
#     rm -rf src

# qemu master
RUN mkdir src && \
    ( cd src && \
    git clone https://git.qemu.org/git/qemu.git && \
    cd qemu && \
    git submodule init && \
    git submodule update --recursive && \
    ./configure --target-list=ppc64le-linux-user --prefix=/milton/qemu && \
    make -j$(nproc) && \
    make -j$(nproc) install && \
    true ) && \
    rm -rf src

# TODO: install mpfr from scratch

COPY run.sh /milton/run.sh

CMD ./run.sh
