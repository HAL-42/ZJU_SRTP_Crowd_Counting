#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Xiaobo Yang
@contact: hal_42@zju.edu.cn
@software: PyCharm
@file: socket_send_recv.py
@time: 2019/5/19 13:02
@desc:
"""
import cv2
import numpy as np
import functools
# import socket
# import time

encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),90]
kHeadLen = 16
kMTU = 1024


def recv_data_len(sock):
    recv = sock.recv(kHeadLen)
    if not recv:
        return None
    return int(recv.decode())


def recv_img(sock, data_len, is_density: bool=False):
    if is_density:
        density_h = int(sock.recv(kHeadLen).decode())
        density_w = int(sock.recv(kHeadLen).decode())
    string_data = b''
    index = 0
    while index + kMTU <= data_len:
        string_data = string_data + sock.recv(kMTU)
        index += kMTU
        sock.send(str(index).ljust(kHeadLen).encode())
    else:
        string_data = string_data + sock.recv(data_len - index)

    if not string_data:
        return None

    #debug
    if data_len != len(string_data):
        print("Error May Occurred at recv_img Function!")
        print("Receive An Img with String Length of: " + str(len(string_data)))
        print("However, the DataLen is: " + str(data_len))
        return None

    if is_density:
        data = np.frombuffer(string_data, dtype='f4').reshape(density_h, density_w)
        return data
    else:
        data = np.frombuffer(string_data, dtype=np.uint8).reshape(data_len, 1)
        return cv2.imdecode(data, 2|4)


recv_density = functools.partial(recv_img, is_density=True)


def send_img(sock, img: np.ndarray, is_density: bool=False):
    if is_density:
        string_data = img.tostring()
    else:
        encoded_img = cv2.imencode('.jpg', img, encode_param)[1]
        string_data = encoded_img.tostring()

    data_len = len(string_data)
    sock.send(str(data_len).ljust(kHeadLen).encode())
    if is_density:
        sock.send(str(img.shape[0]).ljust(kHeadLen).encode())
        sock.send(str(img.shape[1]).ljust(kHeadLen).encode())

    #sock.send(string_data)
    index = 0
    while index + kMTU <= data_len:
        sock.send(string_data[index:index+kMTU])
        # debug
        #print("Send from" + str(index) + '->')
        index += kMTU
        if not sock.recv(kHeadLen):
            print("Can't Receive Confirm pack when index=" + str(index))
            return None
    else:
        sock.send(string_data[index:data_len])
        #debug
        #print("Send from" + str(index) + '->')
    return 0

send_density = functools.partial(send_img, is_density=True)