from bev.main import BEV, default_settings
from romp.utils import WebcamVideoStream
import cv2
import numpy as np
import socket
import time

# socket config
HOST = '127.0.0.1'
PORT = 8000
server_addr = (HOST, PORT)

#s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(server_addr)

def handle_joint_points(joint_points):
    smpl_joints_line = ",".join([str(v) for v in joint_points[:54].flatten()])
    #print(smpl_joints_line)
    return smpl_joints_line

def send_joint_points(joint_points):
    global s
    outdata = handle_joint_points(joint_points)
    #print('sendto ' + str(server_addr) + ': ' + outdata)
    print('sendto ' + str(server_addr))
    #s.sendto(outdata.encode(), server_addr)
    s.send(outdata.encode())

def main():
    settings = default_settings
    settings.mode = 'image'
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    bev_model = BEV(settings)
    # outputs = bev_model(cv2.imread('path/to/image.jpg')) # please note that we take the input image in BGR format (cv2.imread).

    cap = WebcamVideoStream(0)
    cap.start()
    try:
        while True:
            frame = cap.read()
            cv2.imshow('Realtime', frame)
            outputs = bev_model(frame)
            if outputs != None:
                #print(outputs['joints'])
                joint_points = outputs['joints'].astype(np.float16)
                send_joint_points(joint_points)
            else:
                print('No person detected')

            if cv2.waitKey(1) == ord('q'):
                break 

    except KeyboardInterrupt:
        print('shutdown!')
    finally:
        cap.stop()

if __name__ == '__main__':
    main()