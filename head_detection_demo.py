from __future__ import division

import os
import torch as t
from src.config import opt
from src.head_detector_vgg16 import Head_Detector_VGG16
from trainer import Head_Detector_Trainer
from PIL import Image
import numpy as np
from data.dataset import preprocess
import matplotlib.pyplot as plt
import src.array_tool as at
from src.vis_tool import visdom_bbox
import argparse
import src.utils as utils
from tkinter import *
from src.config import opt
import time
from PIL import ImageTk, Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from tkinter import ttk
from random import random
import sys

global prev_poses
prev_poses = []

pos_x_min = 70
pos_x_max = 800
pos_y_max = 108
pos_y_min = 0

res_width = 640
res_height = 480

poses_normalization_thresh = 20


class DisplayPhotoEventHandler(FileSystemEventHandler):
    def __init__(self, canva, watch_file, model_path, head_detector, trainer):
        self.canva = canva
        self.watch_file = watch_file
        self.model_path = model_path
        self.head_detector = head_detector
        self.trainer = trainer

    def on_any_event(self, event):
        st1 = time.time()
        if (not event.is_directory) and (event.event_type == 'closed'):
            try:
                st4 = time.time()
                global prev_poses
                global file
                file = self.watch_file

                file_id = utils.get_file_id(event.src_path)
                img, img_raw, scale = read_img(event.src_path)
                self.trainer.load(self.model_path)
                img = at.totensor(img)
                img = img[None, :, :, :]
                img = img.cuda().float()
                st = time.time()
                pred_bboxes_, _ = self.head_detector.predict(img, scale, mode='evaluate', thresh=0.05)
                et = time.time()
                tt = et - st
                print("[INFO] Head detection over. Time taken: {:.4f} s".format(tt))

                # clear canvas before drawing new circle.
                # self.canva.delete("all")
                # instead of clearing, draw gray circles instead of old values and draw white circles for new values
                for pos in prev_poses:
                    pos_x = translate(pos[0], pos_x_min, pos_x_max, 0, res_width)
                    pos_y = translate(pos[1], pos_y_min, pos_y_max, 0, res_height)

                    self.canva.create_oval((pos_x / scale) - 7, (pos_y / scale) - 7,
                                           ((pos_x / scale) + 7),
                                           ((pos_y / scale) + 7), fill="black", outline="")

                # create current_poses list
                current_poses = []
                for i in range(pred_bboxes_.shape[0]):
                    ymin, xmin, ymax, xmax = pred_bboxes_[i, :]
                    current_pos = (int(xmin), int(ymin))
                    current_poses.append(current_pos)

                current_poses.sort(reverse=True)

                # for pos in current_poses:
                #     self.canva.create_oval((pos[0] / scale), (pos[1] / scale),
                #                            ((pos[0] / scale) + 15),
                #                            ((pos[1] / scale) + 15), fill="white")

                prev_poses.sort(reverse=True)  # keep always ordered list to simulate tracking same person.

                print("********")
                print(prev_poses, current_poses)

                if len(prev_poses) == 0:
                    prev_poses = current_poses.copy()

                # advanced normalization
                if len(current_poses) > 0 and len(prev_poses) > 0:
                    temp_prev_poses = [None] * len(current_poses)
                    temp_current_poses = current_poses.copy()
                    for id_p, ppos in enumerate(prev_poses):
                        for id_c, cpos in enumerate(current_poses):
                            x_thresh_inline = abs(ppos[0] - cpos[0]) < poses_normalization_thresh
                            y_thresh_inline = abs(ppos[1] - cpos[1]) < poses_normalization_thresh
                            if x_thresh_inline and y_thresh_inline:
                                temp_prev_poses[id_c] = ppos
                                temp_current_poses[id_c] = None
                                break

                    for idx in range(len(temp_prev_poses)):
                        pos = temp_prev_poses[idx]
                        if pos is None:
                            temp_prev_poses[idx] = temp_current_poses[idx]

                    prev_poses = temp_prev_poses

                print(prev_poses)
                print("********")

                # new normalization of pos and current_poses

                # bring prev_pos and current_poses to same size.
                # if len(current_poses) < len(prev_pos):
                #     del prev_pos[pred_bboxes_.shape[0] - len(prev_pos)]
                # elif len(current_poses) > len(prev_pos):
                #     # add if too few elements
                #     for i in range(len(current_poses)):
                #         # if more elements are detected, add last in pos history
                #         if len(prev_pos) <= i:
                #             prev_pos.append(current_poses[i])

                et4 = time.time()
                tt4 = et4 - st4
                print("[INFO] BEFORE loop. Time taken: {:.4f} s".format(tt4))

                print("[INFO] {} heads DETECTED.".format(str(len(current_poses))))
                st3 = time.time()
                for i, current_pos in enumerate(current_poses):
                    # uncomment to get position coordinates for pos_x_ and pos_y_
                    # print(current_pos)

                    pos = prev_poses[i]
                    # diff_pos = tuple(map(lambda i, j: i - j, pos, current_pos))

                    # if differences between pos and current_pos are too big, reset pos to current_pos
                    # if diff_pos[0] > 40 or diff_pos[1] > 40:  # TODO: check if same values should be here. perhaps variation is smaller on horizontal and the threshold should be bigger for that case.
                    #     diff_pos = (0, 0)
                    #     pos = current_pos

                    # while pos != current_pos:
                    st2 = time.time()
                    # for q in range(100):  # TODO: externalize variable?
                    #     # print("start loop")
                    #     pos = tuple(map(lambda i, j: i - (j / 100), pos, diff_pos))
                    #
                    #     # pos = tuple(map(lambda i, j, k: i - (j / abs(j)) if i != k and abs(j) != 0 else i,
                    #     #                 pos, diff_pos, current_pos))
                    #
                    #     # translate positions
                    #     # pos_x = translate(pos[0], pos_x_min, pos_x_max, 0, res_width)
                    #     # pos_y = translate(pos[1], pos_y_min, pos_y_max, 0, res_height)
                    #     pos_x = pos[0]
                    #     pos_y = pos[1]
                    #
                    #     # draw path by erasing everything underneath
                    #     self.canva.create_oval((pos_x / scale) - 3, (pos_y / scale) - 3,
                    #                            ((pos_x / scale) + 3),
                    #                            ((pos_y / scale) + 2), fill="black", outline="")
                    #     self.canva.create_oval((pos_x / scale) + random() * 2, (pos_y / scale) + random() * 2,
                    #                            ((pos_x / scale) + 2) + random() * 2,
                    #                            ((pos_y / scale) + 2) + random() * 2, fill="gray", outline="")
                    #
                    #     # TODO: check if we need this logic or just set static sleep. maybe 1/q*10?
                    #     # print("sleep:" + str(abs(0.1 / max(diff_pos))))
                    #     # max_diff = 1 if max(diff_pos) == 0 else max(diff_pos)
                    #     # # print("sleep:" + str(min(abs(0.1 / max_diff), 0.05)))
                    #     # time.sleep(min(abs(0.1 / max_diff), random() / 300))
                    #     # time.sleep(0.001)

                    # translate positions
                    pos_x = translate(pos[0], pos_x_min, pos_x_max, 0, res_width)
                    pos_y = translate(pos[1], pos_y_min, pos_y_max, 0, res_height)
                    # pos_x = pos[0]
                    # pos_y = pos[1]  # TODO: rethink translation if we use only lines.

                    # draw path by erasing everything underneath
                    # self.canva.create_line((pos_x / scale), (pos_y / scale),
                    #                        (current_pos[0] / scale),
                    #                        (current_pos[1] / scale), fill="black", width=3)
                    c_pos_x = translate(current_pos[0], pos_x_min, pos_x_max, 0, res_width)
                    c_pos_y = translate(current_pos[1], pos_y_min, pos_y_max, 0, res_height)
                    self.canva.create_line((pos_x / scale), (pos_y / scale),
                                           (c_pos_x / scale),
                                           (c_pos_y / scale), fill="gray", width=1)

                    et2 = time.time()
                    tt2 = et2 - st2
                    print("[INFO] IF end. Time taken: {:.4f} s".format(tt2))

                    # translate positions
                    pos_x = translate(xmin, pos_x_min, pos_x_max, 0, res_width)
                    pos_y = translate(ymin, pos_y_min, pos_y_max, 0, res_height)
                    # pos_x = pos[0]
                    # pos_y = pos[1]

                    c_pos_x = translate(current_pos[0], pos_x_min, pos_x_max, 0, res_width)
                    c_pos_y = translate(current_pos[1], pos_y_min, pos_y_max, 0, res_height)

                    # 14 is the sice of the circle. this offset draws the circle in the middle
                    self.canva.create_oval((c_pos_x / scale) - 7, (c_pos_y / scale) - 7,
                                           ((c_pos_x / scale) + 7),
                                           ((c_pos_y / scale) + 7), fill="white")

                    utils.draw_bounding_box_on_image_array(img_raw, current_pos[1] / scale, current_pos[0] / scale,
                                                           (current_pos[1] / scale) + 40, (current_pos[0] / scale) + 40)

                    # update previous position for this index.
                    prev_poses[i] = current_pos

                et3 = time.time()
                tt3 = et3 - st3
                print("[INFO] FOR end. Time taken: {:.4f} s".format(tt3))

                st5 = time.time()
                # uncomment this to resize final image.
                # plt.axis('off')
                # plt.imshow(img_raw)
                # # if SAVE_FLAG == 1:
                # # if not os.path.exists(opt.test_output_path):  # Create the directory
                # #     os.makedirs(opt.test_output_path)  # If it doesn't exist
                #
                output_file_path = os.path.join(opt.test_output_path, file_id + '.png')
                # plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)

                im = Image.fromarray(img_raw)
                im.save(output_file_path)

                et5 = time.time()
                tt5 = et5 - st5
                print("[INFO] FILE SAVING. Time taken: {:.4f} s".format(tt5))

                # else:
                #     plt.show()

                local_img = ImageTk.PhotoImage(Image.open(output_file_path))
                # self.label.configure(image=local_img)
                # self.label.image = local_img
                # self.canva.create_oval(5 / scale, 5 / scale, 5 / scale, 5 / scale, fill="black")

            except Exception as e:
                exception_type, exception_object, exception_traceback = sys.exc_info()
                line_number = exception_traceback.tb_lineno
                print(line_number, e)
                pass

            et1 = time.time()
            tt1 = et1 - st1
            print("[INFO] __________ File change trigger over. Time taken: {:.4f} s __________".format(tt1))


def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)


def read_img(path, IM_RESIZE=False):
    f = Image.open(path)
    if IM_RESIZE:
        f = f.resize((res_width, res_height), Image.ANTIALIAS)

    f.convert('RGB')
    img_raw = np.asarray(f, dtype=np.uint8)
    img_raw_final = img_raw.copy()
    img = np.asarray(f, dtype=np.float32)
    # _, H, W = img.shape
    img = img.transpose((2, 0, 1))
    _, H, W = img.shape
    img = preprocess(img)
    _, o_H, o_W = img.shape
    scale = o_H / H
    return img, img_raw_final, scale


def detect(img_path, model_path, watch_file, SAVE_FLAG=0, THRESH=0.01):
    # display points -------------
    root = Tk()
    root.geometry(str(res_width) + 'x' + str(res_height) + '+1200+1200')
    # root.attributes('-fullscreen', True)
    root.title("")
    root.configure(background='red')

    canva = Canvas(root, width=res_width, height=res_height, bd=0, relief='ridge')
    canva.configure(bg='black')
    canva.configure(highlightthickness=0)
    canva.pack()

    head_detector = Head_Detector_VGG16(ratios=[1], anchor_scales=[2, 4])
    trainer = Head_Detector_Trainer(head_detector).cuda()

    try:
        img = ImageTk.PhotoImage(Image.open(img_path))
        # img = ImageTk.PhotoImage(Image.open('projector/1.jpg'))
        # label = ttk.Label(root, image=img, borderwidth=0)
        # label.place(x=10, y=10)

        event_handler = DisplayPhotoEventHandler(canva, watch_file, model_path, head_detector, trainer)
        observer = Observer()
        observer.schedule(event_handler, watch_file, recursive=True)
        observer.start()

        root.mainloop()

    except KeyboardInterrupt:
        pass

    print("-------------------")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    print('end detect')

    # -----------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="test image path")
    parser.add_argument("--model_path", type=str, default='./checkpoints/head_detector_final')
    parser.add_argument("--watch_file", type=str, default='./input/2.jpg')
    args = parser.parse_args()

    global file
    file = args.img_path
    detect(file, args.model_path, args.watch_file, SAVE_FLAG=1)
    # model_path = './checkpoints/sess:2/head_detector08120858_0.682282441835'

    # test_data_list_path = os.path.join(opt.data_root_path, 'brainwash_test.idl')
    # test_data_list = utils.get_phase_data_list(test_data_list_path)
    # data_list = []
    # save_idx = 0
    # with open(test_data_list_path, 'rb') as fp:
    #     for line in fp.readlines():
    #         if ":" not in line:
    #             img_path, _ = line.split(";")
    #         else:
    #             img_path, _ = line.split(":")

    #         src_path = os.path.join(opt.data_root_path, img_path.replace('"',''))
    #         detect(src_path, model_path, save_idx)
    #         save_idx += 1
