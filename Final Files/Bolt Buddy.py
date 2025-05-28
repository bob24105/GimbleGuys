import time								# Python native library
import threading						# Python native library
import numpy as np						# MUST INSTALL NUMPY FIRST
import hebi 							# "pip install hebi-py"
import control.matlab as ctm
from NetFT import Sensor
from scipy import signal
from pynput import keyboard
import cv2								# "pip install opencv-python"
from pupil_apriltags import Detector	# "pip install pupil-apriltags"
import winsound
from playsound import playsound
import simpleaudio as sa





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
KEY_REGIME1 = '1'                    # Return to regime 1



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
    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
    
    




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

# Returns a given vector only if it is above a certain magnitude
# Only works for 2d vectors and scalars with magnitude
# If startFromZero is true, then the vector will start at 0 once it escapes the deadzone
# 	Otherwise, if false, then the vector will start at the edge of the deadzone
def deadzone(vector, magMin, startFromZero):
    try: # Handle the 2d vector case
        vector[0]
        vector = np.array(vector)
        if startFromZero:
            vector = vector - saturate(vector, magMin) if np.linalg.norm(vector) > MINFORCE else np.array([0, 0])
        else:
            vector = vector if np.linalg.norm(vector) > MINFORCE else np.array([0, 0])
            
    except: # Handle the scalar case
        if startFromZero:
            vector = vector - magMin if vector > magMin else vector + magMin if vector < -magMin else 0
        else:
            vector = vector if abs(vector) > magMin else 0
    return vector




# Saturates a vector to a given magnitude
# Works for 2d vectors and scalars with magnitude
def saturate(vector, magMax):
    try: # Handle the 2d vector case
        vector[0]
        # Convert the vector into a numpy array
        vector = np.array(vector)
        
        # Calculate the magnitude of the vector
        total = 0
        for component in vector:
            total += component**2
        mag = total**0.5
        
        
        if mag < magMax:
            return vector
        
        # Scale each component so the magnitude is equal to vmax
        else:
            return vector / (mag / magMax)
    except: # Handle the scalar case
        return magMax if vector > magMax else -magMax if vector < -magMax else vector
        



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



# Returns a vector rotated by a given theta
# Theta in radians
def rotate(vector, theta):
    rotationMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                               [np.sin(theta), np.cos(theta)]])
              
    return np.matmul(rotationMatrix, vector)



# Returns the angle in [rad] of an input vector
def getAngle(vector):
    theta = np.arctan2(vector[1], vector[0])
    return theta

# Convert the raw force readings from the FT sensor into a more useable form
def processRawForce(forceRaw):
    # Define global variables
    global arm_feedback
    
    # Get robot's current position
    arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
    positionRaw = np.array(arm_feedback.position)
    
    # Convert into the proper units
    forceRaw = np.array([x / cpf for x in forceRaw])
    
    # Subtract F0 and WEIGHT off the x and y components
    forceXY = forceRaw[0:2]
    forceXY = forceXY - F0
    
    forceXY = rotate(forceXY, sum(positionRaw))
    forceXY = forceXY - WEIGHT
    
    # Only read the force if it's above MINFORCE
    forceXY = deadzone(forceXY, MINFORCE, False)
    
    # Add the z-force back in
    force = np.array([forceXY[0], forceXY[1], forceRaw[2]])
    
    
    return force



'''
    ###  BEEPZ CODE  ###
# beepz for regime 2A
# beeps after regime 1 switching into regime 2A
def one2twoPitch(frequency, duration, numBeeps, increase):
                #(pitch of beeps, how long the beeps last, how many beeps, how much to increase pitch from the last)    
    for _ in range(numBeeps):
        winsound.Beep(frequency, duration)
        time.sleep(0.001)
        frequency += increase

# beepz for two2one transition
def two2one(frequency, duration, numBeeps, freq2, dur2, numBeeps2):
    for _ in range(numBeeps):
        winsound.Beep(frequency, duration)
        time.sleep(0.001)
    time.sleep(1)
    for _ in range(numBeeps2):
        winsound.Beep(freq2, dur2)
        time.sleep(0.001)


# calibration beepz
def calibrationBeeps(frequency, duration, increase, timeIncrease):
        winsound.Beep(frequency, duration)
        winsound.Beep(frequency, duration)
        winsound.Beep((frequency + increase), duration)
        winsound.Beep(frequency, timeIncrease)

        
        
def play_sound_async(filepath):
    playsound(filepath, block=False)  # block=False makes it asynchronous
 '''

def sa_async(filename):
    try:
        wave_obj = sa.WaveObject.from_wave_file(filename)
        wave_obj.play()
    except Exception as e:
        print("something is wrong with the audio")
        
    

# Convert the raw torque readings from the FT sensor into a more useable form
def processRawTorque(torqueRaw):
    # Convert into proper units
    torqueRaw = np.array([x / cpt for x in torqueRaw])
        
    # Set deadzone for torque
    torque = deadzone(torqueRaw[2], MINTORQUE, True)
    
    return torque
        





''' Define functions for hotkeys '''
# Define the main function that will check for all key presses
def startKeyListeningThread(key):
    if key == KEY_ESTOP:
        eStop()
        return False
    try:
        if key.char == KEY_ZERO:
            recalibrate()
        elif key.char == KEY_REGIME1:
            setRegime1()
    except:
        pass
            
        

### Define the functions for each hot key ###
        
# E-stop to deactivate actuators
stopSignal = threading.Event()
stopSignal.clear()
def eStop():
    stopSignal.set()

# Zero
recalibrateSignal = threading.Event()
recalibrateSignal.clear()
def recalibrate():
    recalibrateSignal.set()
    
# Activate regime 1
regime1Signal = threading.Event()
def setRegime1():
    regime1Signal.set()


### Start the key listening thread ###
keyListeningThread = keyboard.Listener(on_press=startKeyListeningThread)

keyListeningThread.start()



















# MONKEY ALL the cuttoff frequencies need to be changed!!! THey're so high rn so they do absolutely no filtering




''' System Properties '''
### Safety parameters ###
OMEGAMAX = np.pi/2           # [rad/s] Max angular velocity of end effector
VMAX = 9.84                  # [in/s] Max end effector velocity
JOINTCOMPLIANCE = [np.pi/8,  # [rad] Shoulder: max "wiggle room"
                   np.pi/8,  # [rad] Elbow: max "wiggle room"
                   np.pi/8]  # [rad] Wrist: max "wiggle room"
               




### Physical system parameters ###
l1 = 10                      # [in] Link 1 length
l2 = 10                      # [in] Link 2 length
OFFSET = 0                   # [degrees] Orientation of the robot. 0 corresponds to the robot being upright
OFFSET *= 2 * np.pi / 360    # [rad] Converts to radians
EECAMERAOFFSET = 5           # [in] Offset between end effector and camera origin





### FT sensor ###
# The FT sensor returns data in units of [counts]
# The following two values are correction factors
cpf = 1000000                # [counts/lb]
cpt = 1000000                # [counts/lb-in]

HOLDTIME = 3                 # [s] How long the arm will hold its position for before making calibration measurements
CALTHETA = 30                # [degrees] Change in wrist angle between each calibration measurement
CALTHETA *= 2 * np.pi / 360  # [rad] Converts to radians

MINFORCE = 0.5               # [lb] Minimum force that will be detected
MINTORQUE = 5                # [lb-in]

# Calibration values to start with. Comment this section out if you want it to calibrate at the start
F0 = np.array([0.41627094, 3.84162734])   # [lb] The offset of the FT sensor
WEIGHT = rotate([ 0.  ,       -3.15399939], OFFSET)  # [lb] The weight vector of the payload
 
 
 
 
 
### Camera parameters ###
CAMERAINDEX = 0              # Index that specifies one of the total # of connected cameras
TAGSIZE = 0.875              # [in] Side length of April Tag
TAGSIZE *= 0.0254            # [m] Converts to meters





### Singularity parameters ###
THETA2MAX = (9/10) * np.pi   # [rad] Max angle of elbow joint to prevent singularities
THETA2MIN = (1/10) * np.pi   # [rad] Min angle of elbow joint to prevent singularities






### Regime I parameters ###
Ts = 0.03                    # [s] Sample time
LEEWAYMAX = 6                # [in] Max distance between the actual robot position and it's target
VMIN = 0.1				     # [in/s] Velocity deadzone. Any velocity below this magnitude will be ignored
OMEGAMIN = 0.1               # [rad/s] Angular velocity deadzone

## Create controllerR1A transfer function: low pass filter for the input signal ##
K = 4                        # [] Controller gain
omegaC = 10000               # [rad/s] Controller lowpass filter cutoff frequency
lowPass = ctm.tf(K * omegaC, [1, omegaC])
controllerR1A = [cont2sos(lowPass), cont2sos(lowPass)] # Different biquad arrays are needed for the x and y components

## Create controllerR1B transfer function: proportional controller for torque control ##
KT = 0.1                     # [] Controller gain

## Regime I transition parameters ##
VELTRANSITION = .5 * VMAX     	# [in/s] Max end effector velocity for transition to occur
VELDISCONSTANT = 0.6         	# [] Coefficient to convert velocity vector to projected position vector
TRANSITIONDISTANCE = 1       	# [in] Max distance (from tag to projected distance) for transistion to occur
TRANSITIONANGLE = 7.5        	# [degrees] Max angle of tag for transition to occur
TRANSITIONANGLE *= np.pi / 180  # [rad] Convert to radians






### Regime 2 parameters ###
OUTOFVIEWTIME = 0.5          # [s] Amount of time that an april tag can be out of view before transitioning back to regime 1

## Create controllerR2A transfer function: low pass filter for the input signal ##
K = 1                        # [] Gain
omegaC = 10000               # [rad/s] Cutoff frequency for a low pass filter
lowPass = ctm.tf(K * omegaC, [1, omegaC])
controllerR2A = [cont2sos(lowPass), cont2sos(lowPass)] # Different biquad arrays are needed for the x and y components

## Create controllerR2B transfer function: low pass filter for the input signal ##
K = 1                        # [] Gain
omegaC = 10000               # [rad/s] Cutoff frequency for a low pass filter
lowPass = ctm.tf(K * omegaC, [1, omegaC])
controllerR2B = cont2sos(lowPass)

OMEGAMAX2B = np.pi / 8       # [rad/s] Max angular velocity of autonomous regime







### Regime 2C parameters ###
ESCAPEFORCE = 3              # [lb] Force required to escape regime 4






















''' Initialize FT Sensor '''
ip = "192.168.1.50"     # IP address of the FT sensor

# IMPORTANT #
    # Make sure to change your IP address to be:
    # 192.168.1.xxx
    # where xxx is any number other than 50
    # Also set your subnetmask to 255.255.255.0

sensor = Sensor(ip)    # Initialize Sensor object





''' Initialize Hebis '''
# Search for connected Hebis
lookup = hebi.Lookup()
time.sleep(2)

# Create group with three actuators
group = lookup.get_group_from_names("arm", ["shoulder", "elbow", "wrist"])
group.command_lifetime = Ts / 1000 # [ms] The lifetime of each command
000
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
        while not stopSignal.is_set():
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
                newImgSize = (dimensions[0], dimensions[1]))
            

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
                
                # Get tag center
                center = detection.center.astype(int)

                # Get tag corners
                corners = detection.corners # List of coordinates of corners [topleft, top right, bottomright, bottomleft]
                topLeft = np.array(corners[0])
                topRight = np.array(corners[1])
                
                theta = getAngle([topRight[0] - topLeft[0], topRight[1] - topLeft[1]])
                                
                
                # Create new April tag object
                newAprilTag = AprilTag(tag_id, x, y, theta)
                currentTags.append(newAprilTag)
                

                ## Draw information on camera feed ##
                
                # Draw tag outline
                cv2.polylines(undistorted_frame, [corners.astype(int)], True, (0, 255, 0), 2)

                # Draw tag center and ID tag
                cv2.circle(undistorted_frame, tuple(center.astype(int)), 5, (0, 0, 255), -1)
                coord_text = f"Tag {tag_id}: ({pose_t[0]:.2f}, {pose_t[1]:.2f}, {theta:.2f})"
                cv2.putText(undistorted_frame, coord_text, (center[0] + 10, center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                



                
                ## This is for drawing cute little axes on the tags but is not necessary ##
                
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

# Starts a separate camera thread that will check for april tags
cameraThread.start()






























velData = []


''' Define states '''
### Calibration ###
def calibration(tgtPosition): 
    # Define global variables
    global arm_feedback
    global F0
    global WEIGHT
    
    
    sa_async("grubby_mitts.wav")
    #playsound("grubby_mitts.wav", block = False)
    
    
    # Turn off recalibration signal
    recalibrateSignal.clear()
 
    # calibration beeos
    
 
    # Console log
    print("\nCalibrating")
    
    
    
    # Get initial position if there is none
    try:
        if tgtPosition == None:
            arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
            positionRaw = np.array(arm_feedback.position)
            tgtPosition = positionRaw
    except:
        pass

    # Zero sensor
    sensor.zero()
            
            
      
    
    
    ### Collect force readings at 3 angles to find f0 and the weight of the payload ###
    forceReadings = []
    for i in range(3):     
        # Hold position for given HOLDTIME before making measurement
        startTime = time.time()    
        while time.time() - startTime < HOLDTIME and not stopSignal.is_set():
            arm_command.position = tgtPosition
            group.send_command(arm_command)
        
        # Make the measurement
        forceRaw = sensor.force()
        forceRaw = [x / cpf for x in forceRaw]
        
        forceReadings.append(forceRaw[0:2])
        
        # Increase the angle
        if i == 0:
            tgtPosition[2] -= CALTHETA
        elif i ==1:
            tgtPosition[2] += 2 * CALTHETA
        
    
    
    ## Calculate f0 and weight of payload ##
    # Define the 3 points on the arc
    p1 = forceReadings[1]
    p2 = forceReadings[0]
    p3 = forceReadings[2]

    # Find the midpoints between adjacent points
    mid1 = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]  # Midpoint of p1 and p2
    mid2 = [(p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2]  # Midpoint of p2 and p3

    # Find slopes of the perpendicular bisectors
    m1 = -((p2[1] - p1[1]) / (p2[0] - p1[0])) ** -1
    m2 = -((p3[1] - p2[1]) / (p3[0] - p2[0])) ** -1

    # Solve system of equations to find the center of the arc
    # Arc center corresponds to the FT sensor's mean reading with no payload
    A = [[-m1, 1],
         [-m2, 1]]
    b = [[-m1 * mid1[0] + mid1[1]],
         [-m2 * mid2[0] + mid2[1]]]

    F0 = np.squeeze(np.linalg.solve(A, b))


    # Find weight of the payload by finding the radius of the arc
    WEIGHT = np.linalg.norm([p1[0] - F0[0], p1[1] - F0[1]])
    WEIGHT = [0, -WEIGHT] # Convert magnitude into a vector
    WEIGHT = rotate(WEIGHT, OFFSET)
    
       
    
    sa_async("calibration_finished.wav")

    
    
    # Console log
    print("Calibration complete")
    print(f"F0 = {F0}")
    print(f"WEIGHT = {WEIGHT}")
    
    #playsound("calibration_finished.wav", block = False)
    
    return regime1, tgtPosition






































### Regime 1 ###
def regime1(tgtPosition):
    # Define global variables
    global sensor
    global group
    global arm_command
    global arm_feedback
    
    
    
    
    
    
    # Get initial position if there is none
    try:
        if tgtPosition == None:
            arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
            positionRaw = np.array(arm_feedback.position)
            tgtPosition = positionRaw
    except:
        pass
      
    
    
    # Loop to update robot position    
    nextTime = time.perf_counter()
    while not stopSignal.is_set():
        
        ### Transition states when hotkey is pressed ###
        if recalibrateSignal.is_set():
            return calibration, tgtPosition
        elif regime1Signal.is_set():
            return regime1, tgtPosition
        
        
        ### Get current robot state ###
        ## Get force vector ##
        forceRaw = sensor.force()
        force = processRawForce(forceRaw)[0:2] # Only XY forces
        
        ## Get torque vector ##
        torqueRaw = sensor.torque()
        torque = processRawTorque(torqueRaw)
        
        ## Get state of actuators ##
        arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
        positionRaw = np.array(arm_feedback.position)
        
        rHat = getRHat(tgtPosition[0:2]) # Unit vector that points towards the end effector
        
        
        
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
        
        
        
        
        
        ### Convert force/torque input to position control ###
        ## Convert force input to end effector velocity vector ##
        velEE = [cascade(force[i], controllerR1A[i]) for i in range(len(force))]
        
        # Create velocity deadzone
        velEE = deadzone(velEE, VMIN, False)
        
        # Saturate max velocity
        velEE = saturate(velEE, VMAX)
        
        # Convert cartesian velocity to joint velocities
        velJoint = np.matmul(invJacobian(tgtPosition[0], tgtPosition[1]), velEE)
        
        
    
        ## Convert torque input to angular velocity ##
        omegaEE = torque * KT
        
        # Create angular velocity deadzone
        omegaEE = deadzone(omegaEE, OMEGAMIN, False)
        
        # saturate to max angular velocity
        omegaEE = saturate(omegaEE, OMEGAMAX)
        
        
        
        
        ## Convert velocity to position control ##
        tgtPosition[0] += velJoint[0] * Ts
        tgtPosition[1] += velJoint[1] * Ts
        tgtPosition[2] += omegaEE * Ts
        
        # Don't update the target position if the actual robot position is too far from the target position
        if any(abs(tgtPosition - positionRaw) > JOINTCOMPLIANCE):
            tgtPosition[0] -= velJoint[0] * Ts
            tgtPosition[1] -= velJoint[1] * Ts
            tgtPosition[2] -= omegaEE * Ts
        
        

        
        
        
        
        
        
           
        ### Send position control to the robot ###
        arm_command.position = tgtPosition

        group.send_command(arm_command)
        
        
        
            
    
        
        
        ### State Transition Criteria ###
        for tag in visibleAprilTags:
            
            # Check if speed less than max threshold
            if np.linalg.norm(velEE) < VELTRANSITION: 
                tagPos = rotate([tag.x, tag.y], sum(positionRaw)) # Vector from center of camera to tag in Global coordinate frame (x, y)
                
                # MONKEY ADD BEEP NOISES
                
                if abs(tag.theta) < TRANSITIONANGLE:
                    
                    projLoc = velEE * VELDISCONSTANT # Projected distance of end effector based on velocity
                    distanceProjTag = np.linalg.norm([projLoc[0] - tagPos[0], projLoc[1] - tagPos[1]])
                                    
                    # Check if distance between projected location and tag is less than threshold
                    if distanceProjTag < TRANSITIONDISTANCE:
                        return regime2, [tag.id, tgtPosition]
                        
                    
        
        
        
        
        
        
        
        ### Wait until next iteration of clock cycle ###
        nextTime += Ts
        sleepTime = nextTime - time.perf_counter()
        if sleepTime > 0:
            time.sleep(sleepTime)
        else:
            nextTime = time.perf_counter()
            print("Regime 1: clock cycle too slow by", round(-sleepTime * 1000, 2), "ms")
    



























### Regime 2 ###
def regime2(args):
    # Define global variables
    global arm_feedback
    
    #playsound("auto.wav", block = False)
    sa_async("auto.wav")
    
    # Unpack args
    tgtTagID = args[0]
    tgtPosition = args[1]
    
    
    # Get initial position if there is none
    try:
        if tgtPosition == None:
            arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
            positionRaw = np.array(arm_feedback.position)
            tgtPosition = positionRaw
    except:
        pass
    
    
    # Initialize the time of last april tag in frame
    lastSeenTime = time.time()
    
    # Loop to update robot position    
    nextTime = time.perf_counter()
    while not stopSignal.is_set():
        
        ### Transition states when hotkey is pressed ###
        if recalibrateSignal.is_set():
            return calibration, tgtPosition
        elif regime1Signal.is_set():
            return regime1, tgtPosition
        
        
        ### Get current robot state ###
        # Get state of actuators
        arm_feedback = group.get_next_feedback(reuse_fbk=arm_feedback)
        positionRaw = np.array(arm_feedback.position)
        positionXY = cartesian(positionRaw[0:2])
        
        rHat = getRHat(tgtPosition[0:2]) # Unit vector that points towards the end effector
        
        
        
        # Check if target tag is still visible
        if tgtTagID in [tag.id for tag in visibleAprilTags]:
            lastSeenTime = time.time()
            ### Get tag position as input to control system ###
            # Get the April Tag object of our desired tag
            tgtTag = [tag for tag in visibleAprilTags if tag.id == tgtTagID][0]
            
            # Get tag position
            tagPositionRelative = np.array([tgtTag.x, tgtTag.y])
            tagAngleRelative = tgtTag.theta
            
    
            
            
            
            ### Convert tag position input to position control ###
            
            ## Find positional component of the velEE ##
            # Convert tag position vector to velocity vector
            velEEPos = [cascade(tagPositionRelative[i], controllerR2A[i]) for i in range(len(tagPositionRelative))]
            
            # Rotate end effector velocity based on the current angle of the camera
            velEEPos = rotate(velEEPos, positionRaw[0] + positionRaw[1] + positionRaw[2])
            
            
            
            
            ## Find rotational component of the velEE ##

            # Convert tag angle to angular velocity
            omegaEE = cascade(tagAngleRelative, controllerR2B)
                        
            # Saturate max angular velocity
            omegaEE = saturate(omegaEE, OMEGAMAX2B)
            
            # Rotate end effector velocity based on the current angle of the camera
            velEERot = rotate([EECAMERAOFFSET * omegaEE, 0], sum(positionRaw))
            
            
            
            ### Combine the velEE components ###
            velEE = velEEPos + velEERot
            
            # Saturate velEE
            velEE = saturate(velEE, VMAX)
            
            
            
            
            ### Handle interior and boundary cases ###
        
            # Check if at edge of workspace interior or boundary
            atInterior, atBoundary = False, False
            relativeElbowPosition = tgtPosition[1] % (2*np.pi)
            
            if (relativeElbowPosition >= THETA2MAX):
                atInterior = True
            elif (relativeElbowPosition <= THETA2MIN):
                atBoundary = True
            
            # If at interior, delete radial velocities pointing inwards (towards origin)
            if (atInterior):
                if (np.dot(velEE, rHat) < 0): # Check if there is an inward radial velocity and subtract it if so
                    velEE = velEE - (np.dot(velEE, rHat) * rHat)
            
            # If at boundary, delete radial velocities pointing outward
            if (atBoundary):
                if (np.dot(velEE, rHat) > 0): # Check if there is an outward radial velocity and subtract it if so
                    velEE = velEE - (np.dot(velEE, rHat) * rHat)
            
                

            ### Convert cartesian velocity to joint velocities
            
            velJoint = np.matmul(invJacobian(tgtPosition[0], tgtPosition[1]), velEE)
            
            tgtPosition[0] += velJoint[0] * Ts
            tgtPosition[1] += velJoint[1] * Ts
            tgtPosition[2] += omegaEE * Ts
            
            

            ### Send position control to the robot ###

            arm_command.position = [tgtPosition[0], tgtPosition[1], tgtPosition[2]]

            group.send_command(arm_command)
            
            
            
            
        else:
            # Transition back to regime 1 if it has been too long
            if time.time() - lastSeenTime > OUTOFVIEWTIME:
                return regime1, tgtPosition
            
           
            
            
        ### Regime 2 -> Regime 1 Transition ###
        
        # Get raw force vector from FT sensor
        forceRaw = sensor.force()
        forceProcessed = processRawForce(forceRaw)
        force = forceProcessed[0:2]
        forceZ = forceProcessed[2]
        
        if np.linalg.norm(force) > ESCAPEFORCE:      
            return regime1, tgtPosition
            #0two2one(1500, 250, 2, 2000, 500, 1)
        
        
        
        
        
        
        ### Wait until next iteration of clock cycle ###
        nextTime += Ts
        sleepTime = nextTime - time.perf_counter()
        if sleepTime > 0:
            time.sleep(sleepTime)
        else:
            nextTime = time.perf_counter()
            print("Regime 2: clock cycle too slow by", round(-sleepTime * 1000, 2), "ms")
        










































''' Main loop to govern the finite state machine '''
# Define initial state
currState = regime1
args = None


# Start streaming FT sensor data
sensor.startStreaming()
time.sleep(0.1)


### Main loop ###
while not stopSignal.is_set():
    try:
        currState, args = currState(args)
    except:
        pass
    
    # For testing purposes MONKEY
#      currState, args = currState(args)
    

### Exit state ###
# Deactivate Hebis
print("\nDeactivating hebis")
arm_command.position = [np.nan] * 3
group.send_command(arm_command)
print("Hebis deactivated")

# Deactivate FT sensor
print("\nDeactivating FT sensor")
sensor.stopStreaming()
print("FT sensor deactivated")



# Close threads
keyListeningThread.join()
print("\nDeactivating camera")
cameraThread.join()
print("Camera deactivated")



