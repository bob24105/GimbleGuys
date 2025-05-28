import control.matlab as ctm
import numpy as np
from NetFT import Sensor
import time
import threading
import hebi
from scipy import signal
from pynput import keyboard
import cv2
from pupil_apriltags import Detector






''' Formatting conventions '''

''' Heading 1 '''
### Heading 2 ###
## Heading 3 ##
     
   
   
   
    
''' Tasks up next '''
### Miscellaneous ###
# ESTOP to auto shutdown system when max force/torque is reached
# Saturate max joint velocity
# Read into "input shaping" to cancel out vibrations

### Read z force for different things ###
# Testing negative z-force for switching between angle and position adjustment
# Try positive z-force for zeroing new position in regime 2






''' Keyboard Shortcuts '''
KEY_ESTOP = keyboard.Key.esc         # E-stop
KEY_ZERO = '0'                       # Zero FT sensor



''' Classes '''
class Biquad:
    def __init__(self, params):
        self.b0 = params[0]
        self.b1 = params[1]
        self.b2 = params[2]
        self.a0 = params[3]
        self.a1 = params[4]
        self.a2 = params[5]
        self.x0 = 0
        self.x1 = 0
        self.x2 = 0
        self.y1 = 0
        self.y2 = 0
      
      
class AprilTag:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    
    




''' Functions '''
def cascade(xin, biquads):
    y0 = None
    
    for biquad in biquads:
        # Update input
        biquad.x0 = xin
        
        # Calculate output
        y0 = (biquad.b0 * biquad.x0
              + biquad.b1 * biquad.x1
              + biquad.b2 * biquad.x2
              - biquad.a1 * biquad.y1
              - biquad.a2 * biquad.y2) / biquad.a0

        # Update values in current biquad
        biquad.x2 = biquad.x1
        biquad.x1 = biquad.x0
        biquad.y2 = biquad.y1
        biquad.y1 = y0

        # Set output of current biquad as input of next biquad
        xin = y0
        
    return y0


# Convert continuous systems into a discrete biquad cascade
def cont2sos(sys):
    # Convert from matlab transfer function to a useable form
    num, den = ctm.tfdata(sys)
    num = np.squeeze(num)
    den = np.squeeze(den)
    
    # Discretize system
    sysD = signal.cont2discrete((num, den), Ts, method='tustin')
    
    # Convert to second-order sections (SOS)
    sos = signal.tf2sos(sysD[0], sysD[1])
    
    # Convert into an array of Biquads
    biquads = []
    for section in sos:
        biquads.append(Biquad(section))
    
    return biquads



# Saturates a velocity vector at a certain magnitude
def saturate(vel, vMax):
    # Convert the velocity into a numpy array
    vel = np.array(vel)
    
    # Calculate the magnitude of the vel vector
    total = 0
    for component in vel:
        total += component**2
    mag = total**0.5
    
    
    if mag < vMax:
        return vel
    
    # Scale each component so the magnitude is equal to vmax
    else:
        return vel / (mag / vMax)
    



# Returns the inverse Jacobian matrix for given joint angles
# Only works for a 2-linkage system
def invJacobian(theta1, theta2): # [rad] For theta1 and theta2
    # Find Jacobian matrix
    j11 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    j12 = -l2 * np.sin(theta1 + theta2)
    j21 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    j22 = l2 * np.cos(theta1 + theta2)

    jacobian = np.array([[j11, j12],
                         [j21, j22]])
    
    
    return np.linalg.inv(jacobian)



# Returns the position vector in cartesian space that cooresponds to a given set of joint angles
def cartesian(jointPosition):
    return np.array([l1 * np.cos(jointPosition[0]) + l2 * np.cos(jointPosition[0] + jointPosition[1]),
                     l1 * np.sin(jointPosition[0]) + l2 * np.sin(jointPosition[0] + jointPosition[1])])
    


# Returns a unit vector that points in the direction of the end effector
# Only works for a 2-linkage system
def getRHat(jointPosition):
    r = cartesian(jointPosition)
    
    rHat = r / np.linalg.norm(r)
    
    return rHat



# Returns the leeway between the robot's current position and it's target position
# Takes the robot's current and target positions as an array of its joint angles
def getLeeway(actual, tgt):
    leewayVector = cartesian(tgt) - cartesian(actual)
    
    
    
    return np.linalg.norm(leewayVector)


# Returns a vector rotated by a given theta
def rotate(vector, theta):
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
              
    return np.matmul(rotationMatrix, vector)


    





''' Define functions for hotkeys '''
# Define the main function that will check for all key presses
def startKeyListeningThread(key):
    while True:
        if key == KEY_ESTOP:
            ESTOP()
            return False
        

### Define the functions for each hot key ###
# E-stop to deactivate actuators
stopSignal = False
def ESTOP():
    global stopSignal
    stopSignal = True

# Zero

### Start the key listening thread ###
keyListeningThread = keyboard.Listener(on_press=startKeyListeningThread)
keyListeningThread.daemon = True

keyListeningThread.start()
























''' System Properties '''
### Safety parameters ###
OMEGAMAX = 0
VMAX = 9.84                  # [in/s] Max end effector velocity

### Physical system parameters ###
l1 = 10                      # [in] Link 1 length
l2 = 10                      # [in] Link 2 length
OFFSET = 0                   # [degrees] Angle of wrist offset from being level
OFFSET *= 2 * np.pi / 360    # [rad] Converts to radians

### FT sensor ###
MINFORCE = 0.3               # [lb] Minimum force that will be detected

### Camera parameters ###
CAMERAINDEX = 1              # Index that specifies one of the total # of connected cameras
TAGSIZE = 0.875               # [in] Side length of April Tag
TAGSIZE *= 0.0254            # [m] Converts to meters

### Regime I parameters ###
Ts = 0.01                    # [s] Sample time
THETA2MAX = (9/10) * np.pi   # [rad] Max angle of elbow joint to prevent singularities
THETA2MIN = (1/10) * np.pi   # [rad] Min angle of elbow joint to prevent singularities
LEEWAYMAX = 6                # [in] Max distance between the actual robot position and it's target

K = 4                        # [] Gain
omegaC = 10000               # [rad/s] Cutoff frequency for a low pass filter

## Create g1 transfer function: low pass filter for the input signal ##
lowPass = ctm.tf(K * omegaC, [1, omegaC])
g1 = [cont2sos(lowPass), cont2sos(lowPass)] # Different biquad arrays are needed for the x and y components

## Create g2 transfer function: integrator to go from velocity to position ##
integrator = ctm.tf(1, [1, 0])
g2 = [cont2sos(integrator), cont2sos(integrator)]

## Regime I transition parameters

VELTRANSITION = .5*VMAX
VELDISCONSTANT = 0.6 # Constant to convert velocity to distance for projected location
TRANSITIONDISTANCE = 1 # [in] distance (from tag to projected location) for transistion 


### Regime 4 parameters
Ts4 = 0.1                    # [s] Sample time
ESCAPEFORCE = 5              # [lb] Force required to escape regime 4






















''' Initialize FT Sensor '''
ip = "192.168.1.50"     # IP address of the FT sensor

# IMPORTANT #
    # Make sure to change your IP address to be:
    # 192.168.1.xxx
    # where xxx is any number other than 50
    # Also set your subnetmask to 255.255.255.0

# The FT sensor returns data in units of [counts]
# The following two values are correction factors
cpf = 1000000      # [counts/lb]
cpt = 1000000      # [counts/lb-in]

sensor = Sensor(ip)    # Initialize Sensor object





''' Initialize Hebis '''
# Search for connected Hebis
lookup = hebi.Lookup()
time.sleep(2)

# Create group with three actuators
group = lookup.get_group_from_names("arm", ["shoulder", "elbow", "wrist"])
group.command_lifetime = Ts / 1000 # [ms] The lifetime of each command

# Check sucess of group creation
if group is None:
    print("Hebi group not found!")
    exit(1)
else:
    print(f"Hebi connection successful\nConnected to {group.size} actuators")
    
    
# Initialize variable to command our group with
arm_command = hebi.GroupCommand(group.size)

# Initialize variable to recieve feedback from our group with
arm_feedback = hebi.GroupFeedback(group.size)






''' Initialize Camera '''
# Defines the function that will control the camera
# Repeatedly updates the list of april tags that are currently in view
visibleAprilTags = []
def startCameraThread():
    global visibleAprilTags
    
    
    ## Connect to camera ##
    cap = cv2.VideoCapture(CAMERAINDEX)

    # Make sure it opened properly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()
        
        
    ## Define camera parameters ##
    dimensions = [1920, 1080]                          # [px] Dimensions of frame

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, dimensions[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, dimensions[1])


    fx = 1714.45                                       # [px] Focal length in x
    fy = 1627.92                                       # [px] Focal length in y
    cx = dimensions[0] / 2                             # [px] x-coordinate of center
    cy = dimensions[1] / 2                             # [px] y-coordinate of center

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array([0.22703, -2.41431, 0.0, 0.0, 9.04963], dtype=np.float32) # Distortion coeffecients


    ## Create AprilTag detector ##
    detector = Detector('tag36h11')




    ## Capturing live feedback ##
    try:
        while True:
            # Capture current frame
            status, frame = cap.read()
            
            # If frame capture fails
            if not status:
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
            undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

            # Convert to grayscale for AprilTag detection
            gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

            # Detect AprilTags
            results = detector.detect(gray, estimate_tag_pose=True, 
                                      camera_params=[fx, fy, cx, cy], 
                                      tag_size=TAGSIZE)

            # Draw origin (camera's optical center)
            origin = (int(cx), int(cy))
            cv2.circle(undistorted_frame, origin, 10, (0, 0, 255), -1)  # Red circle
            cv2.putText(undistorted_frame, "Origin", (origin[0] + 15, origin[1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            currentTags = []
            for detection in results:
                ## Create an April Tag object
                # Get tag ID
                tag_id = detection.tag_id

                # Get pose information
                pose_R = detection.pose_R            # 3x3 rotation matrix
                pose_t = detection.pose_t.flatten()  # [m] 3x1 translation vector
                pose_t *= 39.3701                    # [in] Convert to inches
                
                # Coordinate sign corrections
                pose_t[1] *= -1
                pose_t[0] *= -1
                x = pose_t[0]  # negative to match hebi coord system
                y = pose_t[1]

                newAprilTag = AprilTag(tag_id, x, y)
                currentTags.append(newAprilTag)
                
                ## Draw information on camera feed ##
                # Get tag center
                center = detection.center.astype(int)

                # Get tag corners
                corners = detection.corners

                # Draw tag outline
                cv2.polylines(undistorted_frame, [corners.astype(int)], True, (0, 255, 0), 2)

                # Draw tag center and ID tag
                cv2.circle(undistorted_frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
                coord_text = f"Tag {tag_id}: ({pose_t[0]:.2f}, {pose_t[1]:.2f})"
                cv2.putText(undistorted_frame, coord_text, (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                



                '''
                This is for drawing cute little axes on the tags but is not necessary
                '''
                # Draw axes
                # axis_length = 0.1  # in meters
                # imgpts, _ = cv2.projectPoints(np.float32([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]]),
                #                             pose_R, pose_t, camera_matrix, dist_coeffs)

                # cv2.line(undistorted_frame, tuple(center.astype(int)), tuple(imgpts[0].ravel().astype(int)), (255, 0, 0), 3)
                # cv2.line(undistorted_frame, tuple(center.astype(int)), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
                # cv2.line(undistorted_frame, tuple(center.astype(int)), tuple(imgpts[2].ravel().astype(int)), (0, 0, 255), 3)



                # Print information
                # print(f"Tag ID: {tag_id}, Position: {pose_t.flatten()}")
                # print(f"Tag Center: {center}")
                # print(f"Position: {pose_t.flatten()}")
                # print(f"Rotation Matrix:\n{pose_R}")
            
            # Update the global list of visible April tags
            visibleAprilTags = currentTags

            # Displaying Live Feedback
            cv2.namedWindow('AprilTag Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('AprilTag Detection', dimensions[0], dimensions[1])
            cv2.imshow('AprilTag Detection', undistorted_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # uses ascii code to find what key is pressed
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    
    
cameraThread = threading.Thread(target=startCameraThread)
cameraThread.daemon = True

# Starts a separate camera thread that will check for april tags
cameraThread.start()














''' Define states '''
### Calibration ###
def calibration(args):
    # Define global variables
    global sensor
    global group
    global arm_command
    global arm_feedback
    global currState
    
    # Stop streaming FT sensor data if it currently is streaming
    try:
        sensor.stopStreaming()
    except:
        pass
         
    # Level the wrist and hold the arm still to accurately tare the sensor
    arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
    positionRaw = np.array(arm_feedback.position)
    tgtPosition = np.array([positionRaw[0], positionRaw[1]])

    for i in range(100):
        arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
        positionRaw = np.array(arm_feedback.position)
        
        arm_command.position = [tgtPosition[0], tgtPosition[1], -(positionRaw[0] + positionRaw[1]) + OFFSET]
        group.send_command(arm_command)
        time.sleep(Ts)

    time.sleep(1)


    # Tare sensor
    sensor.tare()


    # Start streaming FT sensor data
    sensor.startStreaming()
    time.sleep(0.1)
    
    # Update current state
    return regime1, None




### Regime 1 ###
def regime1(args):
    # Define global variables
    global sensor
    global group
    global arm_command
    global arm_feedback
    
    
    
    # Get initial position
    arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
    positionRaw = np.array(arm_feedback.position)
    tgtPosition = np.array([positionRaw[0], positionRaw[1]])
    
    # initalize trigger transition counter
    triggerCount = 0
    
    # Loop to update robot position    
    nextTime = time.perf_counter()
    while not stopSignal:
        ### Get current robot state ###
        # Get raw force vector from FT sensor
        forceRaw = sensor.force()
        forceRaw = np.array([x / cpf for x in forceRaw])
        # print(forceRaw)
        
        # Convert raw force into a more useable form
        force = forceRaw[0:2] # [lb] Input forces in the XY plane
        force = np.array([x if abs(x) > MINFORCE else 0 for x in force])
        force = rotate(force, OFFSET)
        # print(visibleAprilTags)
        
        # Get state of actuators
        arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
        positionRaw = np.array(arm_feedback.position)
        
        rHat = getRHat(tgtPosition) # Unit vector that points towards the end effector
        
        
        
        ### Handle interior and boundary cases ###
        
        # Check if at edge of workspace interior or boundary
        atInterior, atBoundary = False, False
        relativeElbowPosition = tgtPosition[1] % (2*np.pi)
        if (relativeElbowPosition >= THETA2MAX):
            atInterior = True
        elif (relativeElbowPosition <= THETA2MIN):
            atBoundary = True
        
        # If at interior, delete radial forces pointing inwards (towards origin)
        if (atInterior):
            if (np.dot(force, rHat) < 0): # Check if there is an inward radial force and subtract it if so
                force = force - (np.dot(force, rHat) * rHat)
        
        # If at boundary, delete radial forces pointing outward
        if (atBoundary):
            if (np.dot(force, rHat) > 0): # Check if there is an outward radial force and subtract it if so
                force = force - (np.dot(force, rHat) * rHat)
        
        
        
        ### Convert force input to position control ###
        
        # Convert force vector to end effector velocity vector 
        velEE = [cascade(force[i], g1[i]) for i in range(len(force))]
        
        # Cap max velocity of end effector
        velEE = saturate(velEE, VMAX)

        # Convert cartesian velocity to joint velocities
        velJoint = np.matmul(invJacobian(tgtPosition[0], tgtPosition[1]), velEE)
        
        # Convert joint velocities to joint positions
        leeway = getLeeway(positionRaw[0:2], tgtPosition)

        
        tgtPosition += velJoint * Ts
        
        if leeway >= LEEWAYMAX:
            tgtPosition -= velJoint * Ts
        
        
        
        
           
        ### Send position control to the robot ###
        arm_command.position = [tgtPosition[0], tgtPosition[1], -(positionRaw[0] + positionRaw[1]) + OFFSET]

        group.send_command(arm_command)
        
        
        
    
        
        
        ### State Transition Criteria ###
        for tag in visibleAprilTags:
            
            # Check if speed less than max threshold
            if np.linalg.norm(velEE) < VELTRANSITION: 
                tagPos = np.array([tag.x, tag.y])                                      # Vector from center of camera to tag (x, y)
                
                # ADD BEEP NOISES
                
                projLoc = np.array([velEE[0]*VELDISCONSTANT, velEE[1]*VELDISCONSTANT]) # projected distance of end effector based on velocity
                distanceProjTag = np.sqrt((projLoc[0] - tagPos[0])**2 + (projLoc[1] - tagPos[1])**2)
                
                # Check if distance less than max distance threshhold
                if distanceProjTag < TRANSITIONDISTANCE:
                    return regime2, tag.id
                        
                    
        
        
        # if distance between projected location and tag is less than threshold
        
        
        
        
        
        ### Wait until next iteration of clock cycle ###
        nextTime += Ts
        sleepTime = nextTime - time.perf_counter()
        if sleepTime > 0:
            time.sleep(sleepTime)
    


### Regime 2 ###
def regime2(args):
    print("regime 2 initaited")
    while not stopSignal:
        
    return None, None




### Regime 4 ###
def regime4(tgtPosition):
    global sensor
    global group
    global arm_command
    global arm_feedback    
    
    nextTime = time.perf_counter()
    while True:
        ### Command the robot to hold its position ###
        arm_command.position = [tgtPosition[0], tgtPosition[1], -(tgtPosition[0] + tgtPosition[1]) + OFFSET]
        group.send_command(arm_command)
        
        
        
        ### Regime 3 -> Regime 1 Transition ###
        
        # Get raw force vector from FT sensor
        forceRaw = sensor.force()
        forceRaw = np.array([x / cpf for x in forceRaw])
        
        if np.linalg.norm(forceRaw) > ESCAPEFORCE:
            return regime1, None
        
        
        
        
        ### Wait until next iteration of clock cycle ###
        nextTime += Ts4
        sleepTime = nextTime - time.perf_counter()
        if sleepTime > 0:
            time.sleep(sleepTime)
    





''' Main loop to govern the finite state machine '''
# Define initial state
currState = calibration
args = None

### Main loop ###
while not stopSignal:
    currState, args = currState(args)

### Exit state ###
# Deactivate FT sensor
sensor.stopStreaming()

# Deactivate Hebis
arm_command.position = [np.nan] * 3
group.send_command(arm_command)

# Close threads
keyListeningThread.join()










