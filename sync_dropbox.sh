#!/bin/bash

echo "Syncing script between dropboxes - by Tai Duc Nguyen"
echo ""

CONT=""

if [ "$1" = "-u" ]; then
    echo "Syncing Src: /home/sweetbunny/Dropbox/ Dest: C:/User/Tai Duc Nguyen/Dropbox"
    echo -n "Are you sure? (y/n): "
    read CONT
    if [[ "$CONT" = "y" || "$CONT" = "Y" ]]; then
        rsync -r0vzhic --delete --dry-run /home/sweetbunny/Dropbox/ /mnt/C/Users/Tai\ Duc\ Nguyen/Dropbox/ > sync_changes.txt
        cat < sync_changes.txt
        echo -n "Does this look right? (y/n): "
        read CONT
        if [[ "$CONT" = "y" || "$CONT" = "Y" ]]; then
            rsync -rOvzhc --delete --backup --backup-dir="/home/sweetbunny/drbx_bk/up_$(date +\%F_\%H-\%M-\%S)" /home/sweetbunny/Dropbox/ /mnt/C/Users/Tai\ Duc\ Nguyen/Dropbox/
        fi 
    fi
elif [ "$1" = "-d" ]; then
    echo "Syncing Src: C:/User/Tai Duc Nguyen/Dropbox Dest: /home/sweetbunny/Dropbox/"
    echo -n "Are you sure? (y/n): "
    read CONT
    if [[ "$CONT" = "y" || "$CONT" = "Y" ]]; then
        rsync -r0vzhic --delete --dry-run /mnt/C/Users/Tai\ Duc\ Nguyen/Dropbox/ /home/sweetbunny/Dropbox/ > sync_changes.txt
        cat < sync_changes.txt
        echo -n "Does this look right? (y/n): "
        read CONT
        if [[ "$CONT" = "y" || "$CONT" = "Y" ]]; then
            rsync -rOvzhc --delete --backup --backup-dir="/home/sweetbunny/drbx_bk/down_$(date +\%F_\%H-\%M-\%S)" /mnt/C/Users/Tai\ Duc\ Nguyen/Dropbox/ /home/sweetbunny/Dropbox/
        fi        
    fi
else
    echo "Wrong argument. -u for upload and -d for download"
fi

echo "DONE"
