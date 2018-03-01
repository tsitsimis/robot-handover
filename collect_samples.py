"""
Visualize NAO' s camera
capture image samples and label them as good, average or bad for handover
"""

from Robot import Robot
import time
import utils
import cv2
import Image

# initialize the robot
ip_local = "192.168.1.18"
ip_external = "192.168.0.109"
nao = Robot(ip=ip_local, port=9559)
nao.crouch()
nao.init_head()
nao.init_arm()
# nao.init_camera()

# save images to this directory
image_prefix = "/green_"
my_dir = "./new_samples"

cnt = utils.get_counter(my_dir)

# camera = cv2.VideoCapture(1)
while True:
    # read image from camera
    frame = nao.cam2numpy()

    # # object contour
    # found, contour = get_object_contour(frame)
    #
    # if found:
    #     box = contour2box(contour)
    #     cv2.drawContours(frame, [box], 0, (0, 0, 255), 2)

    # show frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Frame", frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        quality = 0
        img = Image.fromarray(frame)
        img.save(my_dir + "/green_" + str(cnt) + "_" + str(quality) + ".jpg")
        cnt += 1

    elif key == ord("2"):
        quality = 50
        img = Image.fromarray(frame)
        img.save(my_dir + image_prefix + str(cnt) + "_" + str(quality) + ".jpg")
        cnt += 1

        nao.open_hand(close=True)
        time.sleep(2)
        nao.open_hand()

    elif key == ord("3"):
        quality = 100
        img = Image.fromarray(frame)
        img.save(my_dir + image_prefix + str(cnt) + "_" + str(quality) + ".jpg")
        cnt += 1


nao.video_proxy.unsubscribe(nao.subscriber)
