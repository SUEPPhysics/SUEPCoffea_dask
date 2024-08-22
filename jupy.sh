#!/bin/bash

PORT=$1
if [ -z "$PORT" ]; then
	PORT=8888
fi

# generate 32 bit random hex string
TOKEN=$(hexdump -n 16 -e '4/4 "%08X" 1 "\n"' /dev/random)

echo "Server url: http://127.0.0.1:${PORT}/?token=${TOKEN}"
echo ""

jupyter notebook --ip 0.0.0.0 --no-browser --notebook-dir . --port $PORT --NotebookApp.token=$TOKEN
