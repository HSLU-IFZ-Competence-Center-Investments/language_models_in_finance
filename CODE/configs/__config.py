import os
from pathlib import Path
from easydict import EasyDict as edict


#%% for DEV

PROJECT_NAME = 'language_models_in_finance' # must be same as project folder name

SUPPORTED_MODELS = []


#%% SETTING WORKSPACE PATHS
WORKSPACE = 'GitHub'  # Currently only 'GitHub'

HOME_DIR = str(Path.home())

GITHUB = os.path.join(HOME_DIR,"GitHub")

if WORKSPACE == "GitHub":
    PROJECT_DIR = os.path.join(GITHUB,PROJECT_NAME)
else:
    raise NotImplementedError(f"Workspace '{WORKSPACE}' not implemented.")


#%% ALL CONFIG DICTS

__C = edict()
cfg = __C

__C.PATH = edict()

__C.PREP = edict()

# ML
__C.DATASET = edict()
__C.MODEL = edict()
__C.OPTIMIZER = edict()
__C.TUNE = edict()
__C.TRAIN = edict()
__C.TEST = edict()


#%% PREAMBLES

__C.PATH.PROJECT = PROJECT_DIR # project folder path
__C.PATH.DATA = os.path.join(PROJECT_DIR,'DATA') # data folder path
__C.PATH.CODE = os.path.join(PROJECT_DIR,'CODE') # code folder path
__C.PATH.OUTPUT = os.path.join(__C.PATH.CODE,'output')
__C.PATH.OUTPUT_PLOT = os.path.join(__C.PATH.OUTPUT,'plots')
__C.PATH.ERR_OUTPUT = os.path.join(__C.PATH.OUTPUT,'errors')
__C.PATH.SDG_OUTPUT = os.path.join(__C.PATH.OUTPUT,'SDG')
__C.PATH.SDG_OUTPUT_AGGREGATED = os.path.join(__C.PATH.OUTPUT,'SDG_aggregated')
__C.PATH.SDG_OUTPUT_AGGREGATED_RECENT_RUNS = os.path.join(__C.PATH.OUTPUT,'Recent_runs')
__C.PATH.SDG_OUTPUT_RANKINGS = os.path.join(__C.PATH.OUTPUT,'ranking')

__C.PATH.PREP = os.path.join(__C.PATH.OUTPUT,'data')
__C.PATH.MODELS = os.path.join(__C.PATH.OUTPUT,'models')
__C.PATH.HTMLS = os.path.join(__C.PATH.DATA,'websites')
__C.PATH.CRAWLER = os.path.join(__C.PATH.DATA,'CRAWLER')

__C.SUPPORTED_MODELS = SUPPORTED_MODELS