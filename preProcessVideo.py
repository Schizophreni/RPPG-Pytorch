
import cv2
import numpy as np
import os
import re
# import matplotlib.pyplot as plt
from multiprocessing import Process

class VideoPreProcess(Process):
    def __init__(self, **kwargs):
        super(VideoPreProcess, self).__init__()  ## 多进程类
        """
        parser args
        video_path: path of video file
        save_video_path: save path of cropped video
        save_npz_path: save path of npz file(Tx68x2 and Tx18)
        label_path: save path of label txt
        video_index: index of video
        videoFPS: fps in gt.csv file
        heartRate: heart rate in video
        mode: train or test
        landmark_path: if mode is test, landmark_path is needed for obtaining landmarks
        """
        self.video_path = kwargs['video_path']
        self.save_video_path = kwargs['save_video_path']
        self.save_npz_path = kwargs['save_npz_path']
        self.label_path = kwargs['label_path']
        self.video_index = kwargs['video_index']
        self.videoFPS = kwargs['fps']
        self.heartRate = kwargs['heartRate']
        self.mode = kwargs.get('mode', 'train')
        self.landmark_path = kwargs.get('landmark_path', '/')
    
    def processVideo(self):
        if self.mode == 'train':
            from lib.core.api.facer import FaceAna
            from lib.core.headpose.pose import get_head_pose, line_pairs
            print('--------------- begin to load model -----------------------')
            facer = FaceAna()
            print('---------------- load model succcess ----------------------')

        print('process video_path ==> %s ' % self.video_path)
        # fil = open(self.label_path, 'a+')
        margin = 25
        crop_H = 400
        crop_W = 400 
        ## open video
        video_capture=cv2.VideoCapture(self.video_path)
        fps = video_capture.get(5) 
        frames = int(video_capture.get(7))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') #opencv3.0
        videoWriter = cv2.VideoWriter(self.save_video_path, fourcc, fps, (crop_W, crop_H))

        RGB_mean = np.zeros((frames, 18)) #  RGB mean value of six ROIs
        landmarksAll = np.zeros((frames, 68, 2))

        for i in range(frames):
            valid_frame = 0
            ret, image = video_capture.read()
            # print(image.shape)
            img_show = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)## img_show is BGR format
            if self.mode == 'train':
                boxes, landmarks, states = facer.run(image) ## image is RGB format
            else:
                self.videoFPS = fps
                self.heartRate = ''
                with open(self.landmark_path, 'r') as fil:
                    lines = fil.readlines()
                line = lines[i].strip().split()
                line = [int(item) for item in line]
                landmarks = np.zeros((1, 68, 2), dtype='int32')
                landmarks[0, :, 0] = line[4::2]
                landmarks[0, :, 1] = line[5::2]

            # assert landmarks.shape[0]==1, 'More than one person or no person in this video'

            crop_img = np.zeros((crop_H, crop_W, 3))
            face_index = 0 ### 默认一张脸
            left_eye = landmarks[face_index][36:42]
            right_eye = landmarks[face_index][42:48]
            nose = landmarks[face_index][27:36]
            face = landmarks[face_index][0:27]

            leftup_eye_h = int(np.min(left_eye[..., 1])) # 左眼睛高度
            leftdown_eye_h = int(np.max(left_eye[..., 1])) # 左眼睛高度
            leftleft_eye_w = int(np.min(left_eye[..., 0])) # 左眼睛左端
            leftright_eye_w = int(np.max(left_eye[..., 0])) # 左眼睛右端

            left_eye_box1 = (leftleft_eye_w, leftdown_eye_h+5)
            left_eye_box2 = (leftright_eye_w, 2*leftdown_eye_h-leftup_eye_h+5)

            rightup_eye_h = int(np.min(right_eye[..., 1]))
            rightdown_eye_h = int(np.max(right_eye[..., 1]))
            rightleft_eye_w = int(np.min(right_eye[..., 0]))
            rightright_eye_w = int(np.max(right_eye[..., 0]))

            right_eye_box1 = (rightleft_eye_w, rightdown_eye_h+5)
            right_eye_box2 = (rightright_eye_w, 2*rightdown_eye_h-rightup_eye_h+5)

            noseup_h = int(np.min(nose[..., 1]))
            nosedown_h = int(np.max(nose[..., 1]))
            noseleft_w = int(np.min(nose[..., 0]))
            noseright_w = int(np.max(nose[..., 0]))

            faceup_h = int(np.min(face[..., 1]))
            facedown_h = min(image.shape[0], int(np.max(face[..., 1]))) ## 防止超出边界，下巴不在视频中也会被预测到
            faceleft_w = int(np.min(face[..., 0]))
            faceright_w = int(np.max(face[..., 0]))

            #start_left = int((crop_W-faceright_w+faceleft_w)/2)
            #start_up = int((crop_H - facedown_h + faceup_h)/2)
            #crop_img[start_up:start_up+facedown_h-faceup_h, start_left:start_left+faceright_w-faceleft_w, :] = img_show[faceup_h:facedown_h, faceleft_w:faceright_w, :]
            #videoWriter.write(crop_img.astype('uint8'))
            window_H = facedown_h - faceup_h
            window_W = faceright_w - faceleft_w ### window_H和window_W是截取的框的大小
            if window_H>window_W:
                ## 截取高度大于宽度，扩大宽度
                window_up_h = faceup_h
                window_down_h = facedown_h
                diff = int((window_H-window_W)/2)
                window_left_w = faceleft_w - diff
                window_right_w = window_left_w + window_H
                window_W = window_H
            elif window_H < window_W:
                ## 截取宽度大于高度，扩大高度
                window_left_w = faceleft_w
                window_right_w = faceright_w
                diff = int((window_W-window_H)/2)
                window_up_h = faceup_h - diff
                window_down_h = window_up_h + window_W
                window_H = window_W
            else:
                window_up_h = faceup_h
                window_down_h =facedown_h
                window_left_w = faceleft_w
                window_right_w = faceright_w
            ### (window_up_h, window_left_w)和(window_down_h, window_right_w)构成截取框
            factor = crop_H/window_H  ## 缩放倍数
            tmp_img = img_show[window_up_h:window_down_h, window_left_w:window_right_w, :].astype('uint8')
            crop_img = cv2.resize(tmp_img, (crop_H, crop_W), 
                                  interpolation=cv2.INTER_CUBIC).astype('uint8') ## cubic插值缩放
            # videoWriter.write(crop_img)
            # plt.imshow(crop_img[..., ::-1])
            # plt.show()

            """
            处理数据
            """
            rela_coordinates = np.zeros((1, 2)) # 由于crop引入的相对坐标
            rela_coordinates[0, 0] = int((window_up_h+window_down_h-crop_H)/2)
            rela_coordinates[0, 1] = int((window_left_w+window_right_w-crop_W)/2)
            # landmarksAll[i, ...] = (landmarks[face_index, :, ::-1] - rela_coordinates)*factor ## 保存相对坐标(h, w)，landmarks为(w, h)
            """
            extract region of interest
            """
            roi_1 = img_show[noseup_h-10:nosedown_h,noseleft_w:noseright_w, ::-1] ## BGR->RGB
            roi_2 = img_show[min(leftup_eye_h, rightup_eye_h):max(leftdown_eye_h, rightdown_eye_h), leftright_eye_w:rightleft_eye_w, ::-1] ## BGR->RGB
            roi_3 = img_show[leftdown_eye_h+5:2*leftdown_eye_h-leftup_eye_h+5, leftleft_eye_w:leftright_eye_w, ::-1] ## BGR->RGB
            roi_4 = img_show[rightdown_eye_h+5:2*rightdown_eye_h-rightup_eye_h+5, rightleft_eye_w:rightright_eye_w, ::-1] ## BGR->RGB
            
            mask_6 = np.zeros(img_show.shape[0:2]) ## region 5的mask
            mask_6[faceup_h:facedown_h, leftleft_eye_w:rightright_eye_w] = 1.0

            mask_face = np.zeros(img_show.shape[0:2]) ## 脸的mask
            land1 = landmarks[face_index][0:17, ...].reshape(1, 17, 2)
            land2 = landmarks[face_index][26:16:-1, ...].reshape(1, 10, 2)
            land = np.concatenate([land1, land2], axis=1).astype('int32') ## 逆时针的脸部landmarks
            cv2.polylines(mask_face, land, 1, 255)
            cv2.fillPoly(mask_face, land, 255)
            mask_face = mask_face/255.0
            
            roi_5 = img_show.transpose((2, 0, 1))[::-1, ...] ##BGR->RGB
            roi_5 = roi_5*mask_face
            roi_5 = roi_5.reshape((3, -1))

            mask_6 = mask_6*mask_face
            roi_6 = img_show.transpose((2, 0, 1))[::-1, ...] ## BGR->RGB
            roi_6 = roi_6*mask_6
            roi_6 = roi_6.reshape((3, -1))
            """
            计算RGB平均值
            """
            RGB_1 = np.mean(np.reshape(roi_1/255.0, (-1, 3)), axis=0)
            RGB_2 = np.mean(np.reshape(roi_2/255.0, (-1, 3)), axis=0)
            RGB_3 = np.mean(np.reshape(roi_3/255.0, (-1, 3)), axis=0)
            RGB_4 = np.mean(np.reshape(roi_4/255.0, (-1, 3)), axis=0)
            RGB_5 = np.sum(roi_5/255.0, axis=1)/np.sum(mask_face)
            RGB_6 = np.sum(roi_6/255.0, axis=1)/np.sum(mask_6)
            RGB = np.concatenate([RGB_1, RGB_2, RGB_3, RGB_4, RGB_5, RGB_6])
            if np.sum(np.isnan(RGB)) == 0: ##valid value
                landmarksAll[valid_frame, ...] = (landmarks[face_index, :, ::-1] - rela_coordinates)*factor ## 保存相对坐标(h, w)，landmarks为(w, h)
                RGB_mean[valid_frame, ...] = RGB ## save valid RGB
                videoWriter.write(crop_img)
                valid_frame += 1

        # videoWriter.release() ## save cropped video
        video_capture.release()
        print('Crop video successfully to ==> %s' % self.save_video_path)
        """
        保存信息
        """
        with open(self.label_path, 'a+') as fil:
            """
            save label
            """
            fil.write('video-%d, ' % self.video_index)
            fil.write(self.save_video_path+', ')
            fil.write(self.save_npz_path+', ')
            fil.write(str(self.videoFPS))
            fil.write(', ')
            fil.write(str(self.heartRate))
            fil.write('\n')
        print('save label ==> %s successfully' % self.label_path)
        np.savez(self.save_npz_path, landmarks=landmarksAll, RGB_mean=RGB_mean)
        print("save npz ==> %s successfully" % self.save_npz_path)
    
    def run(self):
        self.processVideo()
        



