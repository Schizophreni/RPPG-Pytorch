from preProcessVideo import *
import os
import argparse
import pandas as pd
import tensorflow as tf 

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def buildArgs():
    parser = argparse.ArgumentParser(description='settings for video preprocessing')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='/data/Heart-rate/train', \
                        help='root path of video')
    args = parser.parse_args()
    return args
    
def batchProcess(args):
    root_dir = args.root_dir ## root path of data
    save_root = os.path.join(root_dir, 'cropped')
    label_path = os.path.join(save_root, 'labels.txt')
    video_files = os.listdir(root_dir)
    with open(label_path, 'r') as fil:
        lines = fil.readlines()
    exist_file_num = max(0, len(lines)-1)
    begin_idx = exist_file_num//5
    begin_video_idx = exist_file_num % 5+1  ## 已经是子文件夹最后一个视频文件
    if begin_video_idx == 5:
        begin_idx += 1
        begin_video_idx = 1
    else:
        if exist_file_num !=0:
            ## 已经存在文件，需要再加一
            begin_video_idx += 1
    for idx in range(begin_idx, len(video_files)):
        video_paths = os.path.join(root_dir, str(idx+1)) ## 存放视频和gt的文件夹路径
        all_files = os.listdir(video_paths)
        gt_csv_file = os.path.join(video_paths, 'gt.csv')
        # process_list = []
        for video_idx in range(begin_video_idx, len(all_files)):
            video_index = 5*idx+video_idx-1
            kwargs = dict()
            video_path = os.path.join(video_paths, 'video%d.mp4.avi' % video_idx)
            save_video_path = os.path.join(save_root, 'videos/crop-%d.mp4' % video_index) 
            save_npz_path = os.path.join(save_root, 'npzs/crop-%d.npz' % video_index)
            info = pd.read_csv(gt_csv_file).values
            heartRate = info[0, video_idx]
            fps = info[1, video_idx]
            kwargs['video_path'] = video_path
            kwargs['save_video_path'] = save_video_path
            kwargs['save_npz_path'] = save_npz_path
            kwargs['video_index'] = video_index
            kwargs['label_path'] = label_path
            kwargs['fps'] = fps
            kwargs['heartRate'] = heartRate

            processV = VideoPreProcess(**kwargs)
            processV.processVideo()
            # process_list.append(processV)
        #for p in process_list:
         #   p.join()
        # del process_list


def main():
    args = buildArgs()
    batchProcess(args)


if __name__ == '__main__':
    main()