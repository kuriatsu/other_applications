FROM debian

LABEL maintainer="kuriatsu <kuribayashi.atsushi@g.sp.m.is.nagoya-u.ac.jp>"

ENV USERNAME kuriatsu

ENV PASSWORD kuridocker

USER root

WORKDIR /home

RUN echo "now building >>>"

RUN apt update && \
    apt upgrade && \
    apt install -y \
    wget \
    procps python3-gi \
    python3 \
    libatk1.0-0 \
    libcairo2 \
    libglib2.0-0 \
    libgtk-3-0 \
    libpango1.0-0 \
    lsb-release \
    gir1.2-gdkpixbuf-2.0 \
    gir1.2-glib-2.0 \
    gir1.2-gtk-3.0 \
    gir1.2-pango-1.0 && \
    apt --fix-broken install && \
    wget https://linux.dropbox.com/packages/debian/dropbox_2020.03.04_amd64.deb && \
    dpkg -i dropbox_2020.03.04_amd64.deb && \
    rm dropbox_2020.03.04_amd64.deb

RUN apt install sudo && \
    echo "$USERNAME ALL=(ALL) ALL" >> /etc/sudoers.d/$USERNAME && \
    groupadd -g 1000 $USERNAME && \
    useradd -d /home/$USERNAME -m -s /bin/bash -g 1000 -u 1000 $USERNAME && \
    echo "$USERNAME:$PASSWORD" | chpasswd

WORKDIR /home/$USERNAME
