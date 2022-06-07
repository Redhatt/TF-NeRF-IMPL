import os, sys, math
import logging as log
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

# log.basicConfig(level=log.DEBUG)
log.basicConfig(format='%(asctime)s - %(message)s', level=log.INFO)

INFO = log.info
ERROR = log.error
DEBUG = log.debug
WARN = log.warning
CRITICAL = log.critical

BASE_DIR = os.getcwd()
