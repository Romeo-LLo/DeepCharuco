import cv2.aruco as aruco
import numpy as np
import cv2
import xlwt
import os
import csv
def aruco_detection(writer, count):
    # mtx = np.load('./camera_parameters.npy', allow_pickle=True)[()]['mtx']
    # dist = np.load('./camera_parameters.npy', allow_pickle=True)[()]['dist']

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    squareLength = 1.67
    markerLength = 1
    arucoParams = aruco.DetectorParameters_create()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while (True):
        ret, frame = cap.read()

        if ret == True:
            corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, aruco_dict, parameters=arucoParams)

            if np.any(ids != None):
                diamondCorners, diamondIds = aruco.detectCharucoDiamond(frame, corners, ids,
                                                                        squareLength / markerLength)
                if len(diamondCorners) >= 1:
                    count += 1
                    img_fn = '{:04d}.jpg'.format(count)
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imwrite(os.path.join('TrainImage/', img_fn), gray_image)
                    writer.writerow([img_fn, diamondCorners[0][0][0][0], diamondCorners[0][0][0][1],
                                     diamondCorners[0][1][0][0], diamondCorners[0][1][0][1],
                                     diamondCorners[0][2][0][0], diamondCorners[0][2][0][1],
                                     diamondCorners[0][3][0][0], diamondCorners[0][3][0][1]])
                    print(img_fn, ' recorded.')

            # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, mtx, dist)
            # frame = aruco.drawAxis(frame, mtx, dist, rvec, tvec, 1)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            cv2.imshow("Image", frame)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def normal_detection():

    csvFile = './output.csv'
    if os.path.isfile(csvFile):
        with open(csvFile, 'r') as file:
            data = file.readlines()
        lastRow = data[-1]
        count = int(lastRow.split('.')[0])
        with open(csvFile, "a", newline='') as csvFile:
            writer = csv.writer(csvFile)
            aruco_detection(writer, count)
    else:
        with open(csvFile, 'w', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(['Image', 'x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'])
            aruco_detection(writer, 0)

normal_detection()