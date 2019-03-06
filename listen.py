#! /usr/bin/env python
""" Listen for datagram logging message and print them as they are recieved """
import socket
import pickle
from pprint import pprint

UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 9999

serverSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverSock.bind((UDP_IP_ADDRESS, UDP_PORT_NO))

while True:
    data, addr = serverSock.recvfrom(1024)
    dic = pickle.loads(data[4:])
    pprint.pprint(dic)
