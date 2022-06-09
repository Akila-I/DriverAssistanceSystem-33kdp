import argparse
import cv2
import imutils
from object_detection import ObjectDetection
import math
import time
from kalmanfilter import KalmanFilter

# Handle Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to input video")
ap.add_argument("-o", "--output", required=True, help="path to output video")
args = vars(ap.parse_args())

# Initialize Object Detection
od = ObjectDetection()

kf = KalmanFilter()

# Initialize Video Stream Writer
writer = None

# get video stream
cap = cv2.VideoCapture(args["input"])

# try to determine the total number of frames in the video file
try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(cap.get(prop))
    print("[INFO] {} total frames in video".format(total))
# an error occurred while trying to determine the total
# number of frames in the video file
except:
    print("[INFO] could not determine # of frames in video")
    print("[INFO] no approx. completion time can be provided")
    total = -1
# END OF GETTING TOTAL NUMBER OF FRAMES

# Initialize count
count = 0
bottom_mid_points_prev_frame = []

tracking_objects = {}
track_id = 0

# Processing each frame
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break

    # Point current frame
    bottom_mid_points_cur_frame = []

    # danger zone line
    frame_height = frame.shape[0]
    danger_zone_top = int(frame_height*(5/8))
    danger_zone_bottom = int(frame_height*(7/8))

    # cv2.rectangle(frame, (0, danger_line), (frame.shape[1], frame.shape[0]), (255, 0, 0), 1)

    # setting start time of processing a frame
    start = time.time()

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        btx = int((x + x + w) / 2)
        bty = int((y + h))
        bottom_mid_points_cur_frame.append((btx, bty))

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #  Warnings simple logic
        #  #if the rectangle intersecting danger zone lines : give warning

        if y+h > danger_zone_top and y+h < danger_zone_bottom:
            # print("warning")
            cv2.putText(frame, "WARNING", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in bottom_mid_points_cur_frame:
            for pt2 in bottom_mid_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    # tracking_objects[track_id] = pt
                    tracking_objects[track_id] = []
                    tracking_objects[track_id].append(pt)
                    track_id += 1
    else:

        tracking_objects_copy = tracking_objects.copy()
        bottom_mid_points_cur_frame_copy = bottom_mid_points_cur_frame.copy()

        # for object_id, pt2 in tracking_objects_copy.items():
        for object_id, points in tracking_objects_copy.items():
            object_exists = False
            for pt in bottom_mid_points_cur_frame_copy:
                # distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])
                distance = math.hypot(points[-1][0] - pt[0], points[-1][1] - pt[1])

                # Update IDs position
                if distance < 20:
                    #  Warnings prediction logic
                    #  # get the predicted points of the vehicles
                    #  # if the bottom mid point is below danger zone line : give warning

                    # tracking_objects[object_id] = pt
                    predicted = tracking_objects[object_id][0][0], tracking_objects[object_id][0][1]

                    for btm_mid in tracking_objects[object_id]:
                        predicted = kf.predict(btm_mid[0], btm_mid[1])

                    for i in range(10):
                        predicted = kf.predict(predicted[0], predicted[1])
                        if predicted[0] >= 0 and predicted[1] >= 0 and predicted[0] <= frame.shape[0] and predicted[1] <= frame.shape[1]:
                            if predicted[1] > danger_zone_top and predicted[1] < danger_zone_bottom:
                                # print("warning")
                                cv2.putText(frame, "WARNING", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
                        # cv2.putText(frame,str(object_id),predicted,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)

                    cv2.circle(frame, predicted, 5, (255, 0, 0), 2)

                    tracking_objects[object_id].append(pt)
                    object_exists = True
                    if pt in bottom_mid_points_cur_frame:
                        bottom_mid_points_cur_frame.remove(pt)

                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in bottom_mid_points_cur_frame:
            # tracking_objects[track_id] = pt
            tracking_objects[track_id] = []
            tracking_objects[track_id].append(pt)
            track_id += 1

    # for object_id, pt in tracking_objects.items():
    for object_id, points in tracking_objects.items():
        # cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        # cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        cv2.circle(frame, points[-1], 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (points[-1][0], points[-1][1] - 7), 0, 1, (0, 0, 255), 2)

    # setting end time of processing a frame
    end = time.time()

# WRITE TO VIDEO FILE
    if writer is None:
        # initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)
        # some information on processing single frame
        if total > 0:
            elap = (end - start)
            print("[INFO] single frame took {:.4f} seconds".format(elap))
            print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))

    # write the output frame to disk
    writer.write(frame)
# END OF WRITING FRAME

    print("Frames left :", (total-count))

    # Make a copy of the points
    bottom_mid_points_prev_frame = bottom_mid_points_cur_frame.copy()

writer.release()
cap.release()

print("Output video generated")

# References
# https://pysource.com/2021/10/05/object-tracking-from-scratch-opencv-and-python/
# https://pysource.com/2021/11/02/kalman-filter-predict-the-trajectory-of-an-object/

# Command to run
# python3 object_tracking.py --input input.mp4 --output output.avi