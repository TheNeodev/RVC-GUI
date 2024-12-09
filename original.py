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



def extract_model_from_zip(zip_path, output_dir):
    # Extract the folder name from the zip file path
    folder_name = os.path.splitext(os.path.basename(zip_path))[0]

    # Create a folder with the same name as the zip file inside the output directory
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if (member.endswith('.pth') and not (os.path.basename(member).startswith("G_") or os.path.basename(member).startswith("D_")) and zip_ref.getinfo(member).file_size < 200*(1024**2)) or (member.endswith('.index') and not (os.path.basename(member).startswith("trained"))):
                # Extract the file to the output folder
                zip_ref.extract(member, output_folder)

                # Move the file to the top level of the output folder
                file_path = os.path.join(output_folder, member)
                new_path = os.path.join(output_folder, os.path.basename(file_path))
                os.rename(file_path, new_path)

    print(f"Model files extracted to folder: {output_folder}")
    
    
def play_audio(file_path):
    if sys.platform == 'win32':
        audio_file = os.path.abspath(file_path)
        subprocess.call(['start', '', audio_file], shell=True)
    elif sys.platform == 'darwin':
        audio_file = 'path/to/audio/file.wav'
        subprocess.call(['open', audio_file])
    elif sys.platform == 'linux':
        audio_file = 'path/to/audio/file.wav'
        subprocess.call(['xdg-open', audio_file])

def get_full_path(path):
    return os.path.abspath(path)

hubert_model = None
device = config.device
print(device)
is_half = config.is_half

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(
    sid,
    input_audio,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    index_rate,
    crepe_hop_length,
    output_path=None,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model
    if input_audio is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio, 16000)
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )  # 防止小白写错，自动帮他替换掉
     
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            version,
            crepe_hop_length,
            None,
        )
        print(
            "npy: ", times[0], "s, f0: ", times[1], "s, infer: ", times[2], "s", sep=""
        )

        if output_path is not None:
            sf.write(output_path, audio_opt, tgt_sr, format='WAV')

        return "Success", (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    index_rate,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip(
            '"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name)
                         for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                index_rate,
            )
            if info == "Success":
                try:
                    tgt_sr, audio_opt = opt
                    wavfile.write(
                        "%s/%s" % (opt_root, os.path.basename(path)
                                   ), tgt_sr, audio_opt
                    )
                except:
                    info = traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()


# 一个选项卡全局只能有一个音色
def get_vc(weight_root, sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = (weight_root)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": True, "maximum": n_spk, "__type__": "update"}


def clean():
    return {"value": "", "__type__": "update"}


def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True
