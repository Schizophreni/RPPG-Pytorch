from preProcessVideo import *
import os
import argparse 

def buildArgs():
    parser = argparse.ArgumentParser(description='settings for test video processing')
    parser.add_argument('--root_dir', dest='root_dir', type=str, default='/data/HeartRate/test/TestData', help='root path of video')
    parser.add_argument('--mode', dest='mode', type=str, default='test', help='processing mode')
    args = parser.parse_args()
    return args

def testProcess(args):
    root_dir = args.root_dir
    mode = args.mode
    save_root = os.path.join(root_dir, 'cropped')
    label_path = os.path.join(save_root, 'labels.txt')
    video_files = os.listdir(root_dir)
    tot_video_num = len(video_files)-1

    for idx in range(0, tot_video_num//5+1):
        process_list = []
        for video_idx in range(5):
            video_index = 5*idx+video_idx+1
            if video_index > tot_video_num:
                break
            video_paths = os.path.join(root_dir, str(video_index))
            kwargs = dict()
            video_path = os.path.join(video_paths, 'video.avi')
            landmark_path = os.path.join(video_paths, 'landmark.txt')
            save_video_path = os.path.join(save_root, 'videos/crop-%d.mp4' % video_index)
            save_npz_path = os.path.join(save_root, 'npzs/crop-%d.npz' % video_index)
            kwargs['video_path'] = video_path
            kwargs['save_video_path'] = save_video_path
            kwargs['save_npz_path'] = save_npz_path
            kwargs['video_index'] = video_index
            kwargs['label_path'] = label_path
            kwargs['fps'] = 0
            kwargs['heartRate'] = ''
            kwargs['mode'] = args.mode
            kwargs['landmark_path'] = landmark_path
        #print('==============> args', args)
            processV = VideoPreProcess(**kwargs)
            processV.start()
            process_list.append(processV)
        for p in process_list:
            p.join()
        del process_list


def main():
    args = buildArgs()
    testProcess(args)


if __name__ == '__main__':
    main()