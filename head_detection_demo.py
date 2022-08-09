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

global prev_pos
prev_pos = []


class DisplayPhotoEventHandler(FileSystemEventHandler):
    def __init__(self, canva, watch_file, model_path, head_detector, trainer):
        self.canva = canva
        self.watch_file = watch_file
        self.model_path = model_path
        self.head_detector = head_detector
        self.trainer = trainer

    def on_any_event(self, event):
        if (not event.is_directory) and (event.event_type == 'closed'):
            try:
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

                # if less elements are detected than in pos, remove last elements from pos to normalize.
                global prev_pos
                if pred_bboxes_.shape[0] < len(prev_pos):
                    del prev_pos[pred_bboxes_.shape[0] - len(prev_pos)]

                # clear canvas before drawing new circle.
                self.canva.delete("all")
                for i in range(pred_bboxes_.shape[0]):

                    ymin, xmin, ymax, xmax = pred_bboxes_[i, :]
                    current_pos = (int(xmin), int(ymin))

                    print(prev_pos, current_pos)

                    # if more elements are detected, add last in pos history
                    if len(prev_pos) <= i:
                        prev_pos.append(current_pos)

                    pos = prev_pos[i]
                    while pos != current_pos:
                        # print("start loop")
                        self.canva.delete("all")
                        diff_pos = tuple(map(lambda i, j: i - j, pos, current_pos))
                        pos = tuple(map(lambda i, j, k: i - (j / abs(j)) if i != k and abs(j) != 0 else i,
                                        pos, diff_pos, current_pos))

                        prev_pos[i] = pos

                        pos_x = translate(pos[0], 70, 800, 0, 640)
                        pos_y = translate(pos[1], 0, 108, 0, 480)
                        self.canva.create_oval((pos_x / scale) + random() * 2, (pos_y / scale) + random() * 2,
                                               ((pos_x / scale) + 10) + random() * 2,
                                               ((pos_y / scale) + 10) + random() * 2, fill="gray")

                        # print("sleep:" + str(abs(0.1 / max(diff_pos))))
                        max_diff = 1 if max(diff_pos) == 0 else max(diff_pos)
                        # print("sleep:" + str(min(abs(0.1 / max_diff), 0.05)))
                        time.sleep(min(abs(0.1 / max_diff), random()/300))

                    pos_x = translate(xmin, 70, 800, 0, 640)
                    pos_y = translate(ymin, 0, 108, 0, 480)
                    self.canva.create_oval((pos_x / scale), (pos_y / scale),
                                           ((pos_x / scale) + 10),
                                           ((pos_y / scale) + 10), fill="white")

                    utils.draw_bounding_box_on_image_array(img_raw, ymin / scale, xmin / scale, ymax / scale,
                                                           xmax / scale)

                # if (len(pred_bboxes_) != 0):
                #     ymin, xmin, ymax, xmax = pred_bboxes_[0, :]
                #     current_pos = (int(xmin), int(ymin))
                #
                #     print(current_pos)
                #
                #     global pos
                #
                #     # here iterate over previous position and diff position incrementing by +/- 1 to reach current pos.
                #     while pos != current_pos:
                #         # print("start loop")
                #         self.canva.delete("all")
                #         diff_pos = tuple(map(lambda i, j: i - j, pos, current_pos))
                #         pos = tuple(map(lambda i, j, k: i - (j / abs(j)) if i != k and abs(j) != 0 else i,
                #                         pos, diff_pos, current_pos))
                #
                #         pos_x = translate(pos[0], 70, 800, 0, 640)
                #         pos_y = translate(pos[1], 0, 108, 0, 480)
                #         self.canva.create_oval((pos_x / scale) + random() * 2, (pos_y / scale) + random() * 2,
                #                                ((pos_x / scale) + 10) + random() * 2,
                #                                ((pos_y / scale) + 10) + random() * 2, fill="gray")
                #
                #         # print("sleep:" + str(abs(0.1 / max(diff_pos))))
                #         max_diff = 1 if max(diff_pos) == 0 else max(diff_pos)
                #         # print("sleep:" + str(min(abs(0.1 / max_diff), 0.05)))
                #         time.sleep(min(abs(0.1 / max_diff), random()/500))
                #
                #     pos_x = translate(xmin, 70, 800, 0, 640)
                #     pos_y = translate(ymin, 0, 108, 0, 480)
                #     self.canva.create_oval((pos_x / scale), (pos_y / scale),
                #                            ((pos_x / scale) + 10),
                #                            ((pos_y / scale) + 10), fill="white")
                #
                #     utils.draw_bounding_box_on_image_array(img_raw, ymin / scale, xmin / scale, ymax / scale,
                #                                            xmax / scale)

                # uncomment this to resize final image.
                plt.axis('off')
                plt.imshow(img_raw)
                # if SAVE_FLAG == 1:
                if not os.path.exists(opt.test_output_path):  # Create the directory
                    os.makedirs(opt.test_output_path)  # If it doesn't exist

                output_file_path = os.path.join(opt.test_output_path, file_id + '.png')
                plt.savefig(output_file_path, bbox_inches='tight',
                            pad_inches=0)

                # else:
                #     plt.show()

                local_img = ImageTk.PhotoImage(Image.open(output_file_path))
                # self.label.configure(image=local_img)
                # self.label.image = local_img
                # self.canva.create_oval(5 / scale, 5 / scale, 5 / scale, 5 / scale, fill="black")

                # print('end image processing')

            except Exception as e:
                print(e)
                pass


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
        f = f.resize((640, 480), Image.ANTIALIAS)

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
    root.geometry('640x480+1200+1200')
    # # root.attributes('-fullscreen', True)
    # root.title("")
    root.configure(background='red')

    canva = Canvas(root, width=640, height=480, bd=0, relief='ridge')
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
