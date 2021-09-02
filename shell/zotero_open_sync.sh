#!/bin/bash

sudo mount -t cifs -o user=$(whoami),password=$KURINAS_PASSWD,uid=1000,gid=1000 //10.8.0.1/NAS_Documents /mnt/smb_documents/
rsync -auvrt /mnt/smb_documents/zotero/ /home/kuriatsu/Documents/zotero/
rsync -auvrt /home/kuriatsu/Documents/zotero/ /mnt/smb_documents/zotero/
sudo umount /mnt/smb_documents
