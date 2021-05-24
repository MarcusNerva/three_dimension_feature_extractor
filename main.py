import os
import sys
import json
import subprocess
import numpy as np
import torch
from torch import nn
import pickle
import cv2
import PIL.Image as Image
import glob
import h5py

from opts import parse_opts
from model import generate_model
from mean import get_mean
from classify import classify_video


def extract_frames(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        raise Exception('Can not open {}'.format(video_path))

    frames = []
    while True:
        ret, item = cap.read()
        if ret is False:
            break
        img = Image.fromarray(cv2.cvtColor(item, cv2.COLOR_BGR2RGB))
        frames.append(img)
    cap.release()
    return frames


if __name__=="__main__":
    opt = parse_opts()
    opt.mean = get_mean()
    opt.arch = '{}-{}'.format(opt.model_name, opt.model_depth)
    opt.sample_size = 112
    opt.sample_duration = 16
    opt.n_classes = 400
    ext_name = opt.ext
    video_dir = opt.video_dir
    save_dir = opt.save_dir
    dataset_name = opt.dataset_name
    save_path_tpl = os.path.join(save_dir, '{}_{}_{}.{}'.format(dataset_name, opt.arch, '3d', '{}'))

    model = generate_model(opt)
    print('loading model {}'.format(opt.model))
    model_data = torch.load(opt.model)
    assert opt.arch == model_data['arch'], '{} vs {}'.format(opt.arch, model_data['arch'])
    model.load_state_dict(model_data['state_dict'], strict=True)
    model.eval()
    if opt.verbose:
        print(model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    class_names = []
    with open('class_names_list') as f:
        for row in f:
            class_names.append(row[:-1])

    ffmpeg_loglevel = 'quiet'
    if opt.verbose:
        ffmpeg_loglevel = 'info'

    if os.path.exists('tmp'):
        subprocess.call('rm -rf tmp', shell=True)

    video_list = glob.glob(os.path.join(video_dir, '*.{}'.format(ext_name)))

    if opt.mode == 'feature':
        save_path = save_path_tpl.format('hdf5')
        with h5py.File(save_path, 'w') as f:
            for video_path in video_list:
                if not os.path.exists(video_path):
                    print('{} does not exist'.format(video_path))
                    continue
                video_base_path = os.path.basename(video_path)
                video_id = video_base_path.split('.')[0]
                frames = extract_frames(video_path)
                with torch.no_grad():
                    result = classify_video(video_path, frames, video_id, class_names, model, opt)  # List
                result = np.concatenate([item[None, ...] for item in result])
                f[video_id] = result
    else:
        save_path = save_path_tpl.format('pkl')
        outputs = []
        for video_path in video_list:
            if not os.path.exists(video_path):
                print('{} does not exist'.format(video_path))
                continue
            video_base_path = os.path.basename(video_path)
            video_id = video_base_path.split('.')[0]
            frames = extract_frames(video_path)
            with torch.no_grad():
                result = classify_video(video_path, frames, video_id, class_names, model, opt)
            outputs.append(result)
        with open(save_path, 'wb') as f:
            pickle.dump(outputs, f)


