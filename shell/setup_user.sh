#! /bin/bash
apt install sudo
echo "$1 ALL=(ALL) ALL" >> /etc/sudoers.d/$1
groupadd -g 1000 $1
useradd -d /home/$1 -m -s /bin/bash -g 1000 -u 1000 $1
passwd $1
