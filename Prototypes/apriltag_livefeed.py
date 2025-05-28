import cv2
import numpy as np
from pupil_apriltags import Detector
from pynput import keyboard
import threading


class AprilTag:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y




## Camera Setup ##

# Initialize the camera
try: 
    cap = cv2.VideoCapture(1)
    cap.isOpened() == True
except:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()



# Define camera parameters (you'll need to calibrate your camera for accurate results)
dimensions = [1920, 1080]  # pixel dimensions of photo
cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimensions[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimensions[1])

'''These factors come from my mac changing the pixel size of
the images when finding these numbers during calibration'''
mac_res_factor_x = 1 # 1920 / (871.7*2)
mac_res_factor_y = 1 # 1080 / (589.5*2)

fx = 1714.45 * mac_res_factor_x
fy = 1627.92 * mac_res_factor_y
cx = dimensions[0] / 2
cy = dimensions[1] / 2

camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)

dist_vals = [0.22703, -2.41431, 0.0, 0.0, 9.04963]
# dist_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
dist_coeffs = np.array(dist_vals, dtype=np.float32)




## Create AprilTag detector ##
detector = Detector('tag36h11')




## Capturing live feedback ##
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # If frame capture fails
    if not ret:
        print("Error: Could not read frame.")
        break
    
    cx = frame.shape[1] / 2  # Actual width center
    cy = frame.shape[0] / 2  # Actual height center

    # New Camera Matrix to Force Fit for Windows
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (dimensions[0], dimensions[1]),
        alpha = 1,
        newImgSize = (dimensions[0], dimensions[1])
    )

    # Undistort the frame:
    undistored_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Convert to grayscale for AprilTag detection
    gray = cv2.cvtColor(undistored_frame, cv2.COLOR_BGR2GRAY)

    # Detect AprilTags
    results = detector.detect(gray, estimate_tag_pose=True, 
                              camera_params=[fx, fy, cx, cy], 
                              tag_size=4.75/100)  # tag_size in meters

    # Draw origin (camera's optical center)
    origin = (int(cx), int(cy))
    cv2.circle(undistored_frame, origin, 10, (0, 0, 255), -1)  # Red circle
    cv2.putText(undistored_frame, "Origin", (origin[0] + 15, origin[1] + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    aprilTags = []

    for detection in results:

        currentTags = set()

        # Get tag ID
        tag_id = detection.tag_id
        currentTags.add(tag_id)

        # Get tag center
        center = detection.center.astype(int)

        # Get tag corners
        corners = detection.corners


        # Get pose information
        pose_R = detection.pose_R  # 3x3 rotation matrix
        pose_t = detection.pose_t.flatten()  # 3x1 translation vector

        pose_t = pose_t * 39.3701  # converting to inches
        # Coordinate sign corrections
        pose_t[1] *= -1
        x = pose_t[0]
        y = pose_t[1]

        # Checking if april tag exists is the outer list and assigns it to existing if it does exist
        # If not existing is left as None
        existingTag = next((t for t in aprilTags if t.id == tag_id), None)


        if existingTag:  # checks if existingTag is None
            existingTag.x = x
            existingTag.y = y
        else:
            aprilTags.append(AprilTag(tag_id, x, y))

        # Assigns aprilTags list to only what is in the currentTags list
        aprilTags[:] = [tag for tag in aprilTags if tag.id in currentTags]

        # Draw tag outline
        cv2.polylines(undistored_frame, [corners.astype(int)], True, (0, 255, 0), 2)


        # Draw tag center and ID tag
        cv2.circle(undistored_frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
        coord_text = f"Tag {tag_id}: ({pose_t[0]:.2f}, {pose_t[1]:.2f}, {pose_t[2]:.2f})"
        cv2.putText(undistored_frame, coord_text, (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        



        '''
        This is for drawing cute little axes on the tags but is not necessary
        '''
        # Draw axes
        # axis_length = 0.1  # in meters
        # imgpts, _ = cv2.projectPoints(np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]),
        #                             pose_R, pose_t, camera_matrix, dist_coeffs)

        # cv2.line(undistored_frame, tuple(center.astype(int)), tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 3)
        # cv2.line(undistored_frame, tuple(center.astype(int)), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
        # cv2.line(undistored_frame, tuple(center.astype(int)), tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 3)



        # Print information
        print(f"Tag ID: {tag_id}, Position: {pose_t.flatten()}")
        # print(f"Tag Center: {center}")
        # print(f"Position: {pose_t.flatten()}")
        # print(f"Rotation Matrix:\n{pose_R}")
        

    # Displaying Live Feedback
    cv2.namedWindow('AprilTag Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AprilTag Detection', dimensions[0], dimensions[1])
    cv2.imshow('AprilTag Detection', undistored_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # uses ascii code to find what key is pressed
        break
    

cap.release()
cv2.destroyAllWindows()