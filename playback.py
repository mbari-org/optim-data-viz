import os
import cv2
import sys
import glob
import pytz
import pandas
import datetime

import numpy as np

from lrauv_data import LRAUVData

LRAUV_DATA_NAME = 'lrauv_data.csv'

if __name__=="__main__":

    base_path = sys.argv[1]
    lrauv_data = None

    if len(sys.argv) > 2:
        # Try to load lrauvlogs
        lrauv_log_path = sys.argv[2]
        lr = LRAUVData(lrauv_log_path)
        lr.load_all_logs(lrauv_log_path)
        lr.export_2_csv(os.path.join(base_path, LRAUV_DATA_NAME))
        lrauv_data = lr.full_df

    if os.path.exists(os.path.join(base_path,LRAUV_DATA_NAME)):
        lrauv_data = pandas.read_csv(os.path.join(base_path, LRAUV_DATA_NAME))
    


    video_list = sorted(glob.glob(os.path.join(base_path,'*.mp4')))

    video_index = 0

    frame_rate = 30

    # Text properties
    org = (50, 50) # Bottom-left corner of the text string in the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255) # White color
    thickness = 2

    keep_running = True
    while keep_running:
        
        video_path = video_list[video_index]
        cap = cv2.VideoCapture(video_path)

        # Extract start time from file name
        start_time = datetime.datetime.fromisoformat(os.path.basename(video_path).split('_')[1])
        print(start_time)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame,(1920,1080))

            # calculate the frame timestamp
            frame_timestamp = start_time + datetime.timedelta(milliseconds=cap.get(cv2.CAP_PROP_POS_MSEC))
            frame_unixtimestamp = (frame_timestamp - datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)) / datetime.timedelta(seconds=1) + 7*3600 # should not have to add UTC here...hmm

            # Get depth from timestamp
            if lrauv_data is not None:
                query_string = str(frame_unixtimestamp - 0.5) + ' <= unixtime < ' + str(frame_unixtimestamp + 0.5) # A hacky way to find matching time with 1 s window
                rows = lrauv_data.query(query_string)
                output_text = str(frame_timestamp) + ': Depth ' + str(np.mean(rows['depth'])) + ' m'
                cv2.putText(frame, output_text, org, font, font_scale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(frame, str(frame_timestamp), org, font, font_scale, color, thickness, cv2.LINE_AA)
            
            cv2.imshow('Video Playback', frame)
            c = cv2.waitKey(10) & 0xFF
            if c == ord('d'):
                video_index += 1
                if video_index >= len(video_list):
                    video_index = len(video_list) -1
                else:
                    cap.release()
                    video_path = video_list[video_index]
                    cap = cv2.VideoCapture(video_path)
                    # Extract start time from file name
                    start_time = datetime.datetime.fromisoformat(os.path.basename(video_path).split('_')[1])
                    print(start_time)

            if c == ord('a'):
                video_index -= 1
                if video_index < 0:
                    video_index = 0
                else:
                    cap.release()
                    video_path = video_list[video_index]
                    cap = cv2.VideoCapture(video_path)
                    # Extract start time from file name
                    start_time = datetime.datetime.fromisoformat(os.path.basename(video_path).split('_')[1])
                    print(start_time)

            if c == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                keep_running = False
                break

        video_index += 1
        if video_index >= len(video_list):
            break
    

