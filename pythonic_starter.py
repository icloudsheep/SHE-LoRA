#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   pythonic_starter.py
@Time    :   2024/12/05 16:32:24
@Author  :   Jianmin Liu 
@Version :   1.0
@Site    :   https://jianmin.cc
@Desc    :   Pythonic Starter for DEBUG
'''


from flwr.cli.run import run
from pathlib import Path
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if __name__ == "__main__":
    root_path = Path(".")
    run(root_path, federation="local-simulation")
    # run(root_path,federation="local-simulation-gpu")