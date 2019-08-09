from demo_images import load_detect_model, detect_obj

import os
import glob
import cv2
import time
import argparse


def batch_detect(img_dir, detect_model, out_dir):
    fnames = glob.glob(os.path.join(img_dir, '*.jpg'))
    print("Total number of image files =", len(fnames))

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    start_time = time.time()
    for idx, fname in enumerate(fnames):
        if idx >5:
            break
        frame = cv2.imread(fname)
        detect_result = detect_obj(detect_model, frame)

        if detect_result['obj_num'] > 0:
            print('find a frame contains car')
            out_name = fname.replace('.jpg', '_crop.jpg')
            print(detect_result["obj_list"])
            det_res = detect_result["obj_list"][0]
            loc, prob = det_res["location"], det_res["obj_probability"]
            if prob > 0.6:
                x, y, w, h = loc["left"], loc["top"], loc["width"], loc["height"]
                frame_crop = frame[y:y+h, x:x+h]
                cv2.imwrite(os.path.join(out_dir, out_name), frame_crop)
        else:
            continue
        if idx % 100 == 0:
            end_time = time.time()
            avg_time = (end_time - start_time) / 100

            if idx == 100:
                print('Estimated usage time %.4f mins\n' % (avg_time * len(fnames) / 60))


def parse_args():
    parser = argparse.ArgumentParser(description='Extract frames contains need obj')
    parser.add_argument('-i', '--img_dir',  default='/home/gcliu/car_detection/frame_need',
                        type=str, help='Input directory full path')
    parser.add_argument('-o', '--out_dir',  default='/ssd/wfei/data/plate_for_label/hk_entrance/images',
                        type=str, help='Output directory path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    detect_model = load_detect_model()
    print('model load complete......')

    batch_detect(args.img_folder, detect_model)