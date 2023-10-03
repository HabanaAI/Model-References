###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os
import pickle
import socket


def send(sock, data):
    data = pickle.dumps(data)
    rlen = len(data).to_bytes(4, 'big')
    sock.sendall(rlen)
    sock.sendall(data)


def receive_all(sock, length):
    result = b''
    while length > 0:
        data = sock.recv(length)
        if len(data) == 0:
            return None
        result = result + data
        length = length - len(data)
    return result


def receive(sock):
    rlen = receive_all(sock, 4)
    if rlen is None:
        return None
    rlen = int.from_bytes(rlen, 'big')

    data = receive_all(sock, rlen)
    if data is None:
        return None

    return pickle.loads(data)


def connect(socket_path):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    return client


def listen(socket_path):
    if os.path.exists(socket_path):
        os.remove(socket_path)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(socket_path)
    sock.listen()
    return sock
