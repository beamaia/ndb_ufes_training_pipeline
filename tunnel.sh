#!/usr/bin/env bash
# The tunnel from the pod to the MLflow of this laptop, kept alive. Run it on the LAPTOP, in a terminal of its own:
#   caffeinate -dimsu ./tunnel.sh <ip> <port>        (the ip and port of the pod, from the Connect tab)
# When the connection is lost (the Wi-Fi changes, the laptop sleeps) the pod keeps the old session open for a while and port 8002 stays taken, so a new tunnel is refused:
# this script frees the port on the pod and connects again, as many times as it takes. Stop the tunnel you have now before you start it.
set -u
IP=$1
PORT=$2
KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519}
while true; do
    ssh -p "$PORT" -i "$KEY" -o ConnectTimeout=15 "root@$IP" 'kill $(ss -ltnp | grep :8002 | grep -o "pid=[0-9]*" | cut -d= -f2 | sort -u) 2>/dev/null'
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -R 8002:localhost:8000 -p "$PORT" -i "$KEY" "root@$IP"
    echo "$(date +%T) the tunnel closed; trying again in 5 seconds"
    sleep 5
done
