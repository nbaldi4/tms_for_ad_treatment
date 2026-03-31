"""
Visualization, simulation, and signal processing utilities for brain activity modeling
--------------------------------------------------------------------------------------

This module provides a set of helper functions for:
- Visualizing brain surfaces, sensors, and connectivity matrices
- Simulating brain activity using the Jansen–Rit model within The Virtual Brain (TVB)
- Preprocessing EEG and region-based signals
- Computing and plotting functional connectivity and power spectral density (PSD)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score, mean_squared_error
import tvb.simulator.lab as tvb
from tvb.simulator.lab import equations,patterns, monitors
from tvb.simulator.lab import *
from tvb.datatypes import graph
from tvb.datatypes.time_series import TimeSeriesRegion 
from scipy.stats import pearsonr, wasserstein_distance
from scipy.signal import butter, filtfilt, csd, welch
from scipy import signal
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from pymoo.core.termination import Termination
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.projections import  ProjectionSurfaceEEG#,ProjectionMatrix,
from tvb.datatypes.sensors import SensorsEEG
import copy

from classes import JRPSP, Simulator_Modified
from global_variable_creation import get_eeg_monitor, get_conn
import global_variable_creation as gv
from additional_utils import macro_fc

