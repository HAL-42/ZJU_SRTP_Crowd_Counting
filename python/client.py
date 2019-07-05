#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: client.py
@time: 2019/5/18 8:23
@desc:
"""

import socket
import cv2
import numpy as np
import logging
from socket_send_recv import *
from data_preprocession import MinMaxNormalize
import re
import os
import datetime
import glob

from typing import Union

def InitClient(addr: str, port: int) :
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((addr, port))
    except Exception as err:
        print(err)
        print("Can't connect to " + addr + ":" + str(port))
        sock.close()
        return None
    else:
        print("Client has connected to " + addr + ":" + str(port))
        return sock

def SendStream(addr: str, port: int,src: Union[int, str]):
    sock = InitClient(addr, port)
    if not sock:
        raise RuntimeError("Failed to Establish Socket Connect")

    if isinstance(src, str) and (None != re.search('\.jpg', src) or None != re.search('\.png', src)):
        frame = cv2.imread(src, 1)
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("No such Img at " + src)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        send_img(sock, frame)

        cv2.imshow('Client Cam', frame)

        data_len = recv_data_len(sock)
        predict_density = recv_img(sock, data_len, is_density=True)

        # debug
        print(predict_density)

        density_count = np.sum(predict_density)
        print("Current Count=" + str(density_count))
        show_density = MinMaxNormalize(predict_density, 0, 255).astype(np.uint8)
        show_density = cv2.applyColorMap(show_density, cv2.COLORMAP_JET)
        cv2.imshow('Client Received Density Map', show_density)

        file_path, img_name = os.path.split(src)

        if not os.path.isdir(os.path.join(file_path, 'test_result')):
            os.mkdir(os.path.join(file_path, 'test_result'))
        cv2.imwrite(os.path.join(file_path, 'test_result',img_name + "_test.jpg"), show_density)

        with open(os.path.join(file_path, 'test_result.txt'), 'a') as txt_f:
            now = datetime.datetime.now()
            txt_f.write("----------------------------------------------------------------" + '\n')
            txt_f.write("Testing Result Recorded at: " + now.strftime("%Y-%m-%d %H:%M:%S") + '\n')
            txt_f.write("Test img " + img_name + " has density count of:" + str(density_count) + '\n')

        if cv2.waitKey() == 27:
            pass
    else:
        capture = cv2.VideoCapture(src)
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if 0 != send_img(sock, frame):
                break

            cv2.namedWindow('Client Cam', 2)
            cv2.imshow('Client Cam', frame)

            data_len = recv_data_len(sock)
            predict_density = recv_img(sock, data_len, is_density=True)

            # debug
            print(predict_density)

            density_count = np.sum(predict_density)
            print("Current Count=" + str(density_count))
            show_density = MinMaxNormalize(predict_density, 0, 255).astype(np.uint8)
            cv2.namedWindow('Client Received Density Map', 2)
            show_density = cv2.applyColorMap(show_density, cv2.COLORMAP_JET)
            cv2.imshow('Client Received Density Map', show_density)

            if cv2.waitKey(100) == 27:
                break
    sock.close()
    if src == 0:
        print("Connection End By User")
    else:
        print("Connection End for Video has been Send Out")
    cv2.destroyAllWindows()


def TestDir(dir_path):
    for img_path in glob.glob(os.path.join(dir_path, '*.jpg')):
        SendStream('10.13.71.169', 12345, img_path)
    for img_path in glob.glob(os.path.join(dir_path, '*.png')):
        SendStream('10.13.71.169', 12345, img_path)


if __name__ == '__main__':
    try:
        print("Start Sending Stream")
        # TestDir('..\\dataset\\My_Test')
        SendStream('10.13.71.169', 12345, '..\\dataset\\My_Test\\tq.jpg')
        # SendStream('10.13.71.169', 12345, '..\\dataset\\My_Test\\yb1.mp4')
        # SendStream('10.13.71.169', 12345, '..\\dataset\\UCF_CC_50\\1.jpg')
        # SendStream('10.13.71.169', 12345, 0)
    except RuntimeError as err:
        logging.error(err)
    except Exception as err:
        logging.error("Unkown Error")
        logging.error(err)
    finally:
        print("Connection End")
