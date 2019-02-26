#!/bin/bash

PID="$(ps -s | grep "[p]ssh" | awk '{print $2}')"
# [p] is to prevent grep from picking up the actual grep process itself
kill -s 9 $PID
