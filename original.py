import random
import string
import soundfile as sf


import os
import sys
import torch
import warnings

now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
os.makedirs(os.path.join(now_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "output"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)

from vc_infer_pipeline import VC
from fairseq import checkpoint_utils
from scipy.io import wavfile
from my_utils import load_audio
from infer_pack.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from infer_pack.modelsv2 import SynthesizerTrnMs768NSFsid_nono, SynthesizerTrnMs768NSFsid
from multiprocessing import cpu_count
import threading
from time import sleep
from time import sleep
import traceback
import numpy as np
import subprocess
import zipfile
from config import Config

config = Config()
