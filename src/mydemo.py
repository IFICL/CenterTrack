from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import csv
import os
import sys
import cv2
import json
from tqdm import tqdm
import glob
import copy
import numpy as np
from opts import opts
from detector import Detector

# CUDA_VISIBLE_DEVICES=0 python mydemo.py tracking --videolist=0
# For videolist 0: processed up to 4/8
# For videolist 1: processed up to 5/8
# For videolist 2: processed up to 
# For videolist 3: processed up to 3/7


def create_dirlist(dirlist, opt):
    if opt.videolist == 0:
        out = dirlist[4:]
    elif opt.videolist == 1:
        out = dirlist[5:]
    elif opt.videolist == 2:
        # out = dirlist[5:]
        print('This video list has processed!')
        sys.exit()
    elif opt.videolist == 3:
        out = dirlist[3:]
    return out



# image_ext = ['jpg', 'jpeg', 'png', 'webp']
# video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge', 'display']


def demo(opt, idx, dirname):
    detector = Detector(opt)

    is_video = False
    # Demo on images sequences
    if isinstance(opt.demo, list):
        image_names = opt.demo
    else:
        image_names = [opt.demo]

    # Initialize output video
    out = None
    # if opt.save_video:
    #     fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #     out = cv2.VideoWriter('../results/{}.mp4'.format(
    #         opt.exp_id + '_' + out_name), fourcc, opt.save_framerate, (
    #         opt.video_w, opt.video_h))

    if opt.debug < 5:
        detector.pause = False
    cnt = 0
    results = {}

    while True:
        if is_video:
            _, img = cam.read()
            if img is None:
                break
        else:
            if cnt < len(image_names):
                img = cv2.imread(image_names[cnt])
            else:
                break
        cnt += 1

        # resize the original video for saving video results
        if opt.resize_video:
            img = cv2.resize(img, (opt.video_w, opt.video_h))

        # skip the first X frames of the video
        if cnt < opt.skip_first:
            continue

        # cv2.imshow('input', img)

        # track or detect the image.
        ret = detector.run(img)
        # log run time
        time_str = 'frame {} |'.format(cnt)
        for stat in time_stats:
            time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
        # print(time_str)

        # results[cnt] is a list of dicts:
        #  [{'bbox': [x1, y1, x2, y2], 'tracking_id': id, 'category_id': c, ...}]
        if opt.filter:
            results[cnt] = filtering(ret['results'])
        else:
            results[cnt] = ret['results']
        # import pdb; pdb.set_trace()
        # save debug image to video
        if opt.save_video:
            if is_video:
                out.write(ret['generic'])
            else:
                cv2.imwrite('../results/demo{}.jpg'.format(cnt), ret['generic'])

        # esc to quit and finish saving video
        if cv2.waitKey(1) == 27:
            break

    save_and_exit(opt, out, results, dirname, idx)


def filtering(res):
    #  [person, bicycle, car, motorcycle, bus, train, truck]
    interested_class = [1, 2, 3, 4, 6, 7, 8]
    new_res = []
    for obj in res:
        if obj['class'] in interested_class:
            new_res.append(obj)
    
    return new_res



def save_and_exit(opt, out=None, results=None, dirname=None, idx=None):
    if opt.save_results and (results is not None):
        save_dir = os.path.join(dirname, str(idx).zfill(8) + '.json')
        tqdm.write('saving results to {}'.format(save_dir))
        json.dump(_to_list(copy.deepcopy(results)),
                open(save_dir, 'w'))
    if opt.save_video and out is not None:
        out.release()
    # sys.exit(0)


def _to_list(results):
    for img_id in results:
        for t in range(len(results[img_id])):
            for k in results[img_id][t]:
                if isinstance(results[img_id][t][k], (np.ndarray, np.float32)):
                    results[img_id][t][k] = results[img_id][t][k].tolist()
    return results

def process():
    opt = opts().init()
    # opt.task = 'tracking'
    # opt.tracking = True
    opt.dataset = 'coco'
    opt.load_model = '../models/coco_tracking.pth'
    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.save_results = True
    opt.save_video = False
    opt.message = False
    # Set 0 to cancel visualization
    opt.debug = 0
    opt.filter = True
    data_folder = '../binaural-tour/img_audio'
    video_len = 5   # 5 seconds
    dirlist = os.listdir(data_folder)
    dirlist.sort()
    count = 8
    dirlist = dirlist[opt.videolist * count: min((opt.videolist+1) * count, len(dirlist))]
    # to continue
    dirlist = create_dirlist(dirlist, opt)
    # import pdb; pdb.set_trace()
    for dirname in tqdm(dirlist, desc="Video Processing"):
        datapath = os.path.join(data_folder, dirname)
        csv_path = os.path.join(datapath, 'meta.csv')
        frame_path = os.path.join(datapath, 'frames')

        mot_path = os.path.join(datapath, 'mot')
        if not os.path.exists(mot_path):
            os.mkdir(mot_path)
        
        # load csv
        meta = []
        for row in csv.reader(open(csv_path, 'r'), delimiter=','):
            meta.append(row)
        fps = float(meta[2][0].split(':')[-1][1:-1])
        frame_len = int(fps * video_len)

        read_list = glob.glob('%s/*.jpg' % frame_path)
        read_list.sort()

        video_clip_num = int(len(read_list) // frame_len)
        for i in tqdm(range(0, video_clip_num + 1), desc="Sub Video Processing"):
            image_list = read_list[i * frame_len: min((i + 1) * frame_len, len(read_list))]
            # import pdb; pdb.set_trace()
            opt.demo = image_list
            demo(opt, i, mot_path)
    






if __name__ == '__main__':
    process()
