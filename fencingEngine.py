from math import sqrt
import subprocess
import shlex
import shutil
import os
import numpy as np
import cv2
import json
import glob


BODY_PARTS = [ "Nose","Neck","RShoulder","RElbow","RWrist", "LShoulder","LElbow", "LWrist", "MidHip", "RHip","RKnee","RAnkle","LHip", "LKnee","LAnkle","REye","LEye", "REar", "LEar","LBigToe", "LSmallToe", "LHeel","RBigToe", "RSmallToe"]
# source https://theailearner.com/2018/10/15/extracting-and-saving-video-frames-using-opencv-python/

if os.path.isdir('openpose'):
    os.chdir('openpose')


def save_frames_to_JSON_file(path):
    # video file
    cap = cv2.VideoCapture(path)
    i = 0

    while cap.isOpened:
        ret, frame = cap.read()
        if ret == False:
            break
        filename = 'fencing' + str(i) + '.jpg'
        if i < 10:
            filename = 'fencing0' + str(i) + '.jpg'
        cv2.imwrite(filename, frame)
        i += 1
    cap.release()
    cv2.destroyAllWindows()


def get_frames(video_path, file_path, rescale=True):
    i = 1
    # video file
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    people_frame = []

    while cap.isOpened:
        ret, frame = cap.read()

        if ret == False:
            break
        if rescale:
            frame = rescale_frame(frame, 50)
        if i < 10:
            cv2.imwrite(file_path + '00' + str(i) + '.jpg', frame)
        elif i < 100:
            cv2.imwrite(file_path + '0' + str(i) + '.jpg', frame)
        else:
            cv2.imwrite(file_path + str(i) + '.jpg', frame)
        people_frame.append((frame, cap.get(cv2.CAP_PROP_POS_MSEC)))

        i += 1
    # 'openpose/output/frames/frame'
    cap.release()
    cv2.destroyAllWindows()

    return people_frame


def rescale_frame(frame, percent=75):
    height, width, layers = frame.shape
    dim = (320, 180)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def get_people_keypoints(json_files, people_frame, timestamp):
    # Loop through all json files in output directory
    # Each file is a frame in the video
    i = 0
    height, width, layers = people_frame[0].shape
    boundary = (height * 0.15)
    people_keypoints = dict()
    temp_frame = []
    temp_timestamp = []
    for file in json_files:

        temp_df = json.load(open(file))
        temp = []
        N = 3

        people = temp_df['people']
        if len(people) >= 2:
            subList1 = [people[0]["pose_keypoints_2d"][n:n + N][:2] for n in
                        range(0, len(people[0]["pose_keypoints_2d"]), N)]
            subList2 = [people[1]["pose_keypoints_2d"][n:n + N][:2] for n in
                        range(0, len(people[1]["pose_keypoints_2d"]), N)]

            max1 = 0
            max2 = 0

            y_subList1 = get_y_center(subList1)
            y_subList2 = get_y_center(subList2)

            if y_subList1 - boundary <= y_subList2 <= y_subList1 + boundary:
                max1, max2, subList1, subList2 = get_max_and_list(subList1, subList2)

                set_keypoints_and_frames(max1, max2, subList1, subList2, people_keypoints, temp_frame, people_frame[i],
                                         timestamp[i], temp_timestamp)

            elif len(people) == 3:
                subList3 = [people[2]["pose_keypoints_2d"][n:n + N][:2] for n in
                            range(0, len(people[2]["pose_keypoints_2d"]), N)]
                y_subList3 = get_y_center(subList3)

                if y_subList1 - boundary <= y_subList3 <= y_subList1 + boundary:
                    max1, max2, subList1, subList2 = get_max_and_list(subList1, subList3)
                else:
                    max1, max2, subList1, subList2 = get_max_and_list(subList2, subList3)
                set_keypoints_and_frames(max1, max2, subList1, subList2, people_keypoints, temp_frame, people_frame[i],
                                         timestamp[i], temp_timestamp)

        i += 1

    return temp_frame, people_keypoints, temp_timestamp


def contains_zero(sublist):
    for i, l in enumerate(sublist):
        if (2 <= i <= 7 or 10 <= i <= 11 or 13 <= i <= 14) and l[0] == 0:
            return True
    return False


def set_keypoints_and_frames(max1, max2, subList1, subList2, people_keypoints, temp_frame, people_frame, timestamp,
                             temp_timestamp):
    if max1 > max2:
        set_keypoints('right', subList1, people_keypoints)
        set_keypoints('left', subList2, people_keypoints)
    else:
        set_keypoints('right', subList2, people_keypoints)
        set_keypoints('left', subList1, people_keypoints)
    temp_frame.append(people_frame)
    temp_timestamp.append(timestamp)


def get_max_and_list(list1, list2):
    max1 = get_max_x(list1)
    max2 = get_max_x(list2)
    return max1, max2, list1, list2


def set_keypoints(id, subList, people_keypoints):
    temp = people_keypoints.get(id, [])
    temp.append(subList)
    people_keypoints[id] = temp


def swap_keypoints_max_right(keypoints, point_list=[(2, 4), (9, 11)]):
    for subList in keypoints:
        for points in point_list:

            for i in range(points[0], points[1] + 1):
                temp = max(subList[i], subList[i + 3])
                subList[i + 3] = min(subList[i], subList[i + 3])
                subList[i] = temp


def swap_keypoints_max_left(keypoints, point_list=[(2, 4), (9, 11)]):
    for subList in keypoints:
        for points in point_list:
            for i in range(points[0], points[1] + 1):

                temp = min(subList[i], subList[i + 3])
                subList[i + 3] = max(subList[i], subList[i + 3])
                subList[i] = temp


def get_max_x(subList):
    res = list(zip(*subList))
    max_x = max(res[0])
    return max_x


def get_y_center(subList):
    res = list(zip(*subList))
    y = [i for i in res[1] if i != 0]
    average = (min(y) + max(y)) / 2
    return average


def front_leg_left(keypoint, left, right):
    if keypoint[left] > keypoint[right]:
        return 'left'
    return 'right'


def front_leg_right(keypoint, left, right):
    if keypoint[left] < keypoint[right]:
        return 'left'
    return 'right'


def move_first(left_keypoints, right_keypoints, LToe, RToe, timestamp):
    x_value = 10
    y_value = 1
    r_point = right_keypoints[0][BODY_PARTS.index(RToe)]
    l_point = left_keypoints[0][BODY_PARTS.index(LToe)]

    for i in range(1, len(left_keypoints)):
        if left_keypoints[i][BODY_PARTS.index(LToe)][0] == 0 or right_keypoints[i][BODY_PARTS.index(RToe)][0] == 0:
            continue
        if left_keypoints[i][BODY_PARTS.index(LToe)][0] - l_point[0] >= x_value:
            if r_point[1] - right_keypoints[i][BODY_PARTS.index(RToe)][1] > 5 and l_point[1] - \
                    left_keypoints[i][BODY_PARTS.index(LToe)][1] <= y_value:
                return 'left', round(timestamp[i] / 1000, 2), i

        if r_point[0] - right_keypoints[i][BODY_PARTS.index(RToe)][0] >= x_value:
            if r_point[1] - right_keypoints[i][BODY_PARTS.index(RToe)][1] <= y_value and l_point[1] - \
                    left_keypoints[i][BODY_PARTS.index(LToe)][1] >= 5:
                return 'right', round(timestamp[i] / 1000, 2), i

    return None


def calculate_slope(start_point, end_point):

    if (int(start_point[0]) != int(end_point[0])):
        return (int(end_point[1]) - int(start_point[1])) // (int(end_point[0]) - int(start_point[0]))
    return None


def distance(p0, p1):
    if p0[0] == 0 or p1[0] == 0:
        return None
    return sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def calculate_angle_abc(a, b, c):
    AB = distance(a, b)
    BC = distance(b, c)
    AC = distance(a, c)
    from math import acos, degrees
    if AB != None and BC != None and AC != None and AB * BC != 0:
        return int(degrees(acos((AB * AB + BC * BC - AC * AC) / (2.0 * AB * BC))))
    else:
        return None


def calculate_speed(distance, time):
    return distance / time


def attacker(keypoint_left, keypoint_right, timestamp, len_blade, RShoulder, RWrist, LShoulder, LWrist):
    prev = 0
    for i in range(1, len(keypoint_left)):
        if keypoint_left[i][BODY_PARTS.index('MidHip')][0] == 0 or keypoint_right[i][BODY_PARTS.index('MidHip')][
            0] == 0:
            # print(keypoint_left[i][BODY_PARTS.index('MidHip')])
            continue

        if keypoint_left[i][BODY_PARTS.index(LShoulder)][0] == 0 or keypoint_left[i][BODY_PARTS.index(LWrist)][
            0] == 0 or keypoint_right[i][BODY_PARTS.index(RShoulder)][0] == 0 or \
                keypoint_right[i][BODY_PARTS.index(RWrist)][0] == 0:
            prev += 1
            continue

        if int(keypoint_right[i][BODY_PARTS.index(RShoulder)][0] - keypoint_left[i][BODY_PARTS.index(LWrist)][
            0]) <= len_blade or int(
                keypoint_right[i][BODY_PARTS.index(RWrist)][0] - keypoint_left[i][BODY_PARTS.index(LShoulder)][
                    0]) <= len_blade:

            speed_left = calculate_speed(
                keypoint_left[i][BODY_PARTS.index('MidHip')][0] - keypoint_left[prev][BODY_PARTS.index('MidHip')][0],
                timestamp[i] - timestamp[prev])
            speed_right = calculate_speed(
                keypoint_right[prev][BODY_PARTS.index('MidHip')][0] - keypoint_right[i][BODY_PARTS.index('MidHip')][0],
                timestamp[i] - timestamp[prev])

            if speed_left > 0 and speed_right > 0 and abs(speed_left - speed_right) <= 0.03:
                return 'Simultaneous'
            if speed_left > speed_right:
                return 'left'
            elif speed_left < speed_right:
                return 'right'
        prev += 1


def techniques(keypoint_left, keypoint_right, LElbow, LWrist, LShoulder, RElbow, RWrist, RShoulder, len_blade,
               people_frames,timestamp):
    attack = attacker(keypoint_left, keypoint_right, timestamp, len_blade, RShoulder, RWrist,
                      LShoulder, LWrist)
    point, index = got_point(keypoint_left, keypoint_right, LElbow, LWrist, LShoulder, RElbow, RWrist, RShoulder,
                             len_blade, people_frames,timestamp)
    remove_directories()
    if attack == 'Simultaneous':
        return {'technique': attack, 'point': 'none'}
    if attack == 'right' and point == 'left':
        return {'technique': 'Parry Riposte', 'point': 'Left'}
    if attack == 'left' and point == 'right':
        return {'technique': 'Parry Riposte', 'point': 'Right'}
    return {'technique': 'Attack', 'point': attack}


def got_point(keypoint_left, keypoint_right, LElbow, LWrist, LShoulder, RElbow, RWrist, RShoulder, len_blade,
              people_frames, timestamp):
    percentage = 0.20
    attack = attacker(keypoint_left, keypoint_right, timestamp, len_blade, RShoulder, RWrist,
                      LShoulder, LWrist)
    if attack == 'Simultaneous':
        return attack, 0
    for i in range(len(keypoint_right) - 1, 0, -1):

        if keypoint_left[i][BODY_PARTS.index(LShoulder)][0] == 0 or keypoint_left[i][BODY_PARTS.index(LElbow)][
            0] == 0 or keypoint_left[i][BODY_PARTS.index(LWrist)][0] == 0 or \
                keypoint_right[i][BODY_PARTS.index(RShoulder)][0] == 0 or keypoint_right[i][BODY_PARTS.index(RElbow)][
            0] == 0 or keypoint_right[i][BODY_PARTS.index(RWrist)][0] == 0:
            continue
        if keypoint_left[i][BODY_PARTS.index(LShoulder)][0] < keypoint_right[i][BODY_PARTS.index(RWrist)][0]:

            left_slope_elbow_shoulder = calculate_slope(keypoint_left[i][BODY_PARTS.index(LElbow)],
                                                        keypoint_left[i][BODY_PARTS.index(LShoulder)])
            right_elbow_shoulder = calculate_slope(keypoint_right[i][BODY_PARTS.index(RElbow)],
                                                   keypoint_right[i][BODY_PARTS.index(RShoulder)])
            left_slope_elbow_wrist = calculate_slope(keypoint_left[i][BODY_PARTS.index(LElbow)],
                                                     keypoint_left[i][BODY_PARTS.index(LWrist)])
            right_slope_elbow_wrist = calculate_slope(keypoint_right[i][BODY_PARTS.index(RElbow)],
                                                      keypoint_right[i][BODY_PARTS.index(RWrist)])
            dist_left = int(
                keypoint_right[i][BODY_PARTS.index(RShoulder)][0] - keypoint_left[i][BODY_PARTS.index(LWrist)][0])
            dist_right = int(
                keypoint_right[i][BODY_PARTS.index(RWrist)][0] - keypoint_left[i][BODY_PARTS.index(LShoulder)][0])

            if left_slope_elbow_shoulder != None and 0 <= left_slope_elbow_shoulder <= 1:
                if left_slope_elbow_wrist != None and -1 <= left_slope_elbow_wrist <= 1:
                    if int(len_blade * percentage) <= dist_left <= len_blade and dist_left < dist_right:
                        cv2.imshow(people_frames[i])
                        return 'left', i
            if right_elbow_shoulder != None and -1 <= right_elbow_shoulder <= 0:
                if right_slope_elbow_wrist != None and -1 <= right_slope_elbow_wrist <= 1:
                    if int(len_blade * percentage) <= dist_right <= len_blade:
                        cv2.imshow(people_frames[i])
                        return 'right', i


def last_frame_attacker(people_keypoints, dist_left, dist_right):
    for i in range(len(people_keypoints['left']) - 1, 0, -1):
        if attacker(people_keypoints['left'][i], people_keypoints['right'][i], dist_left, dist_right) != None:
            return i


def draw_poly_line(img, pts, color_select=(255, 0, 0), thick=2):
    poly_line_thickness = thick
    poly_closed = False

    cv2.polylines(img, np.int32([pts]), poly_closed, color_select, thickness=poly_line_thickness)


def draw_line(img, start_point, end_point, label, color_select=(255, 0, 0), thick=2):
    image = cv2.line(img, start_point, end_point, color_select, thick)


def run_openpose_image():
    command_line = 'build/examples/openpose/openpose.bin --image_dir output/frames --display 0  --write_json output/Json --render_pose 0  --number_people_max 3'
    args = shlex.split(command_line)
    p = subprocess.run(args)

def remove_directories():
    shutil.rmtree('output/Json')
    shutil.rmtree('output/frames')


def create_directories():
    if not os.path.isdir('output/Json'):
        os.mkdir('output/Json')
    if not os.path.isdir('output/frames'):
        os.mkdir('output/frames')


def get_frames_json_files(file_path, root_path='.'):
    create_directories()
    frame_path = 'output/frames/'

    # Paths - should be the folder where image was stored
    path_to_image_file = root_path + "/" + file_path
    people_frames, timestamp = zip(*get_frames(path_to_image_file, frame_path))
    files = sorted(glob.glob("output/frames/*.jpg"))
    run_openpose_image()
    json_files = sorted(glob.glob("output/Json/*.json"))

    return people_frames, json_files, timestamp


def get_frames_keypoints(file_path, root_path=''):
    people_frames, json_files, timestamp = get_frames_json_files(file_path, root_path=root_path)
    people_frames, people_keypoints, timestamp = get_people_keypoints(json_files, people_frames, timestamp)

    return people_frames, people_keypoints, timestamp


def get_point(point):
    point = [int(x) for x in point]
    return tuple(point)


def get_values(people_frames, people_keypoints):
    Lp2 = 'LShoulder'
    Lp1 = 'LElbow'
    Rp2 = 'RShoulder'
    Rp1 = 'RElbow'
    Llist = []
    RList = []

    if front_leg_left(people_keypoints['left'][0], BODY_PARTS.index("LKnee"), BODY_PARTS.index("RKnee")) == 'left':
        start_index_left = BODY_PARTS.index(Lp1)
        end_index_left = BODY_PARTS.index(Lp2)
        swap_keypoints_max_left(people_keypoints['left'])
        Llist = ['LElbow', 'LWrist', 'LShoulder', 'LKnee', 'LAnkle', 'LBigToe']
    else:
        start_index_left = BODY_PARTS.index(Rp1)
        end_index_left = BODY_PARTS.index(Rp2)
        swap_keypoints_max_right(people_keypoints['left'])
        Llist = ['RElbow', 'RWrist', 'RShoulder', 'RKnee', 'RAnkle', 'RBigToe']

    if front_leg_right(people_keypoints['right'][0], BODY_PARTS.index("LKnee"), BODY_PARTS.index("RKnee")) == 'right':
        start_index_right = BODY_PARTS.index(Rp2)
        end_index_right = BODY_PARTS.index(Rp1)
        swap_keypoints_max_left(people_keypoints['right'])
        Rlist = ['RElbow', 'RWrist', 'RShoulder', 'RKnee', 'RAnkle', 'RBigToe']
    else:
        start_index_right = BODY_PARTS.index(Lp1)
        end_index_right = BODY_PARTS.index(Lp2)
        swap_keypoints_max_right(people_keypoints['right'])
        Rlist = ['LElbow', 'LWrist', 'LShoulder', 'LKnee', 'LAnkle', 'LBigToe']

    len_blade = int((people_keypoints['right'][0][BODY_PARTS.index(Rlist[3])][0] -
                     people_keypoints['left'][0][BODY_PARTS.index(Llist[3])][0]) * 88 / 400)

    return Llist, Rlist, start_index_left, end_index_left, start_index_right, end_index_right, len_blade


def draw_frames(people_keypoints, people_frames):
    Llist, Rlist, start_index_left, end_index_left, start_index_right, end_index_right, _ = get_values(people_frames,
                                                                                                       people_keypoints)
    for i in range(len(people_keypoints['left'])):
        draw_line(people_frames[i], get_point(people_keypoints['left'][i][start_index_left]),
                  get_point(people_keypoints['left'][i][end_index_left]), 'left')  # blue
        draw_line(people_frames[i], get_point(people_keypoints['right'][i][start_index_right]),
                  get_point(people_keypoints['right'][i][end_index_right]), 'right', (0, 0, 255))  # red
        cv2.imshow(people_frames[i])


def get_technique(file_path):
    people_frames, people_keypoints, timestamp = get_frames_keypoints(file_path, root_path='.')
    Llist, Rlist, start_index_left, end_index_left, start_index_right, end_index_right, len_blade = get_values(
        people_frames, people_keypoints)

    return techniques(people_keypoints['left'], people_keypoints['right'], Llist[0], Llist[1], Llist[2], Rlist[0],
                      Rlist[1], Rlist[2], len_blade, people_frames,timestamp)
