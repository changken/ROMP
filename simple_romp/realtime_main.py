import bev
from romp.utils import WebcamVideoStream
import cv2

def handle_joint_points(joint_points):
    smpl_joints_line = ",".join([str(v) for v in joint_points[:54].flatten()])
    print(smpl_joints_line)


def main():
    settings = bev.main.default_settings
    settings.mode = 'image'
    # settings is just a argparse Namespace. To change it, for instance, you can change mode via
    # settings.mode='video'
    bev_model = bev.BEV(settings)
    # outputs = bev_model(cv2.imread('path/to/image.jpg')) # please note that we take the input image in BGR format (cv2.imread).

    cap = WebcamVideoStream(0)
    cap.start()
    try:
        while True:
            frame = cap.read()
            outputs = bev_model(frame)
            if outputs != None:
                #print(outputs['joints'])
                joint_points = outputs['joints']
                handle_joint_points(joint_points)
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