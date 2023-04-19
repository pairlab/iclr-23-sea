#!/bin/sh

kill -9 $(pgrep -f "python polyhydra.py")
kill -9 $(pgrep -f "python -m torchbeast.polyhydra")
