import time
import socket



UDP_IP_ADDRESS = "127.0.0.1"
UDP_PORT_NO = 27000
Message = b"Start Unity"

clientSock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

clientSock.sendto(Message, (UDP_IP_ADDRESS, UDP_PORT_NO))

print("Starting Unity for Editor Training with Debugging ...")