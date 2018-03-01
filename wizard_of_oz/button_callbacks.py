from naoqi import ALProxy
import vision_definitions as vd
import almath
import time
from time import gmtime, strftime
import numpy as np
import Image


# ttsProxy = ALProxy("ALTextToSpeech", IP, PORT)
# joint names
headYawName = "HeadYaw"
headPitchName = "HeadPitch"
RShoulderRollName = "RShoulderRoll"
RShoulderPitchName = "RShoulderPitch"
RElbowRollName = "RElbowRoll"
RElbowYawName = "RElbowYaw"
RWristYawName = "RWristYaw"
RHandName = "RHand"

audio_path = "/home/nao/audioAthena/"   # girl
# audio_path = "/home/nao/audio/"       # boy

shoulder_pitch_max = -30


def nao_stiffness(ip, port, v):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.setStiffnesses(["Body"], v)


# postures: Sit, Stand, Crouch
def nao_posture(ip, port, posture, speed=0.5):
    posture_proxy = ALProxy("ALRobotPosture", ip, port)
    posture_proxy.goToPosture(posture, speed)  # blocking call


def nao_walk(ip, port, dist):
    motion_proxy = ALProxy("ALMotion", ip, port)
    speed = 0.2
    motion_proxy.setAngles("RHand", 0.0, speed)

    motion_proxy.moveTo(dist, 0.0, 0.0)  # blocking call, moveTo(x, y, theta)


def nao_move_arm(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    speed = 0.2
    motion_proxy.setAngles(RShoulderRollName, -80 * almath.TO_RAD, speed)
    time.sleep(1)
    motion_proxy.setAngles(RShoulderPitchName, 4 * almath.TO_RAD, speed)
    time.sleep(1)
    motion_proxy.setAngles(RShoulderRollName, -2 * almath.TO_RAD, speed)
    time.sleep(1)
    speed = 0.05
    motion_proxy.setAngles(RElbowRollName, 5.3 * almath.TO_RAD, speed)
    motion_proxy.setAngles(RShoulderRollName, 9.8 * almath.TO_RAD, speed)
    motion_proxy.setAngles(RShoulderPitchName, shoulder_pitch_max * almath.TO_RAD, speed)
    motion_proxy.setAngles(RElbowYawName, 62.8 * almath.TO_RAD, speed)
    motion_proxy.setAngles(RWristYawName, -74.0 * almath.TO_RAD, 0.5)
    time.sleep(1)
    speed = 1.0
    motion_proxy.setAngles(RHandName, 0.8, speed)


def nao_unmove_arm(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    speed = 0.2

    motion_proxy.setAngles(RShoulderRollName, -80 * almath.TO_RAD, speed)
    time.sleep(1)
    motion_proxy.setAngles(RShoulderPitchName, 100 * almath.TO_RAD, speed)
    time.sleep(1)
    motion_proxy.setAngles(RShoulderRollName, -16 * almath.TO_RAD, speed)
    time.sleep(1)
    motion_proxy.setAngles(RElbowYawName, 115.0 * almath.TO_RAD, speed)
    motion_proxy.setAngles(RWristYawName, -28.0 * almath.TO_RAD, 0.5)


def nao_close_hand(ip, port, speed=0.5):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.setAngles(RHandName, 0.0, speed)  # non-blocking call


def nao_open_hand(ip, port, speed=0.8):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.setAngles(RHandName, 1.0, speed)  # non-blocking call


def nao_close_open_hand(ip, port, n, speed=0.8):
    motion_proxy = ALProxy("ALMotion", ip, port)
    delay = 0.2
    for i in range(n):
        motion_proxy.setAngles(RHandName, 0.0, speed)  # non-blocking call
        time.sleep(delay)
        motion_proxy.setAngles(RHandName, 1.0, speed)  # non-blocking call
        time.sleep(delay)


def nao_turn_wrist(ip, port, val, speed=0.8):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.setAngles(RWristYawName, val, speed)  # non-blocking call


def nao_head_yaw(ip, port, yaw=40, pitch=-20):
    motion_proxy = ALProxy("ALMotion", ip, port)
    speed = 0.2
    motion_proxy.setAngles("HeadYaw", yaw * almath.TO_RAD, speed)  # non-blocking call
    motion_proxy.setAngles("HeadPitch", pitch * almath.TO_RAD, speed)  # non-blocking call
    time.sleep(2)
    motion_proxy.setAngles("HeadYaw", 0, speed)  # non-blocking call
    motion_proxy.setAngles("HeadPitch", 0, speed)  # non-blocking call
    time.sleep(1)


def nao_head_hand(ip, port, n_head, n_hand):
    motion_proxy = ALProxy("ALMotion", ip, port)
    head_speed = 0.2
    for i in range(n_head):
        motion_proxy.setAngles("HeadYaw", 40 * almath.TO_RAD, head_speed)  # non-blocking call
        motion_proxy.setAngles("HeadPitch", -20 * almath.TO_RAD, head_speed)  # non-blocking call
        time.sleep(1)
        nao_close_open_hand(ip, port, n_hand)
        time.sleep(1)
        motion_proxy.setAngles("HeadYaw", 0, head_speed)  # non-blocking call
        motion_proxy.setAngles("HeadPitch", 0, head_speed)  # non-blocking call
        time.sleep(1)


def nao_lower_arm(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    hand_speed = 0.2
    motion_proxy.setAngles(RShoulderPitchName, 58 * almath.TO_RAD, hand_speed)  # non-blocking call


def nao_lift_arm(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    hand_speed = 0.2
    motion_proxy.setAngles(RShoulderPitchName, shoulder_pitch_max * almath.TO_RAD, hand_speed)


def nao_lift_arm_point(ip, port):
    motion_proxy = ALProxy("ALMotion", ip, port)
    hand_speed = 0.2
    motion_proxy.setAngles(RShoulderPitchName, -12 * almath.TO_RAD, hand_speed)
    motion_proxy.setAngles(RElbowRollName, 21 * almath.TO_RAD, hand_speed)

    time.sleep(5)

    motion_proxy.setAngles(RElbowRollName, 5.3 * almath.TO_RAD, hand_speed)
    motion_proxy.setAngles(RShoulderPitchName, shoulder_pitch_max * almath.TO_RAD, hand_speed)


def nao_say(ip, port, file):
    audio_proxy = ALProxy("ALAudioPlayer", ip, port)
    audio_proxy.playFile(audio_path + file + ".wav")


def nao_open_camera(ip, port):
    video_recorder_proxy = ALProxy("ALVideoRecorder", ip, port)
    filename = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    video_recorder_proxy.startRecording("/home/nao/camera/recordings/", filename)


def nao_close_camera(ip, port):
    video_recorder_proxy = ALProxy("ALVideoRecorder", ip, port)
    n_frames, path = video_recorder_proxy.stopRecording()
    print n_frames, path


def nao_bend(ip, port, target):
    postureProxy = ALProxy("ALRobotPosture", ip, port)
    motion_proxy = ALProxy("ALMotion", ip, port)
    # Send NAO to Pose Init
    # postureProxy.goToPosture("StandInit", 0.5)

    motion_proxy.wbEnable(True)

    # Example showing how to set orientation target for LArm tracking.
    effectorName = "RArm"

    motion_proxy.wbEnableEffectorControl(effectorName, True)
    time.sleep(2.0)
    if target == 1:
        target_coordinate = [0.2, 0.0, 0.8]  # bend slightly to the front
    elif target == 2:
        target_coordinate = [0.2, 0.2, 0.0]  # bend down

    motion_proxy.wbSetEffectorControl(effectorName, target_coordinate)

    time.sleep(2.0)
    motion_proxy.wbEnable(False)


def get_ref_image(compass_proxy):
    nao_image = compass_proxy.getReferenceImage()
    image_width = nao_image[0]
    image_height = nao_image[1]
    array = nao_image[6]

    # convert to numpy array
    frame = Image.frombytes("L", (image_width, image_height), array)
    frame = np.array(frame)
    return frame


def get_current_image(compass_proxy):
    nao_image = compass_proxy.getCurrentImage()
    image_width = nao_image[0]
    image_height = nao_image[1]
    array = nao_image[6]

    # convert to numpy array
    frame = Image.frombytes("L", (image_width, image_height), array)
    frame = np.array(frame)
    return frame


def nao_walk_compass(ip, port, d, open_cam=True):
    # turn-on camera
    if open_cam:
        try:
            nao_open_camera(ip, port)
            print "camera on!"
        except:
            print "camera off"

    # visual compass
    process_name = "VisualCompassTest"

    compass_proxy = ALProxy("ALVisualCompass", ip, port)
    memory_proxy = ALProxy("ALMemory", ip, port)
    motion_proxy = ALProxy("ALMotion", ip, port)

    # set reference image
    compass_proxy.enableReferenceRefresh(True)
    compass_proxy.setResolution(1)

    compass_proxy.subscribe(process_name)
    # ref_image = get_ref_image(compass_proxy)

    dist = 0
    while dist <= d:
        # Get the deviation information from the ALMemory event.
        try:
            deviation = memory_proxy.getData("VisualCompass/Deviation")
            # match_info = memory_proxy.getData("VisualCompass/Match")

            # wy = deviation[0][0]
            wz = deviation[0][1]

            # if len(wz) > 0:
            # walk
            dx = 0.1
            motion_proxy.moveTo(dx, 0.0, -wz)
            dist += dx
        except:
            continue

    # unsubscribe
    compass_proxy.unsubscribe(process_name)


def nao_turn(ip, port, theta):
    motion_proxy = ALProxy("ALMotion", ip, port)
    motion_proxy.moveTo(0.0, 0.0, theta)


