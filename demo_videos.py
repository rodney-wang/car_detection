from demo_images import load_detect_model, detect_obj
import matplotlib.pyplot as plt
import os
import glob
import cv2
import time
import argparse


def detect_videos(video_fp, detect_model, frame_per_second_extract=2, frame_need_dir):
    souce = video_fp
    cap = cv2.VideoCapture(souce)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Video: fps: {}, frames: {}, time: {.4f} mins'.format(fps, frames, frames/fps))
    print('Extract {} frame per second'.format(frame_per_second_extract))
    
    frame_count = 0
    frame_step = fps // frame_per_second_extract
    start_time_ori = time.time()
    start_time = time.time()
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        frame_count += 1

        if frame_count % frame_step != 0:
            continue

        detect_result = detect_obj(detect_model, frame)

        if detect_result['obj_num'] > 0:
            print('find a frame contains car')
            cv2.imwrite(os.path.join(frame_need_dir, str(time.time()) + '.jpg'), frame)

        if frame_count % 100 ==0:
            end_time = time.time()
            avg_time = (end_time - start_time) / 100
            
            if frame_count == 100:
                print('Estimated usage time %.4f mins\n' % (avg_time*frames/60))
    
            print('%d frames have been processed, avg time is %.4f' % (frame_count, avg_time))
            start_time = time.time()


    print('time using is %.4f' % (time.time() - start_time_ori))
    
def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames contains need obj')
    parser.add_argument('--video_fp', type=str, help='Input video file full path')
    parser.add_argument('--frame_per_second_extract', type=int, default=2, help='The frames count per second extract')
    parser.add_argument('--frame_need_dir', type=str, help='Extract frames contains need obj saved dir')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    detect_model = load_detect_model()
    print('model load complete......')
    
    detect_videos(args.video_fp, detect_model, args.frame_per_second_extract, frame_need_dir)