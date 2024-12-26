#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh && \
conda activate tracesim && \
jupyter notebook --no-browser --ip="*" --port=9779 --NotebookApp.token=777