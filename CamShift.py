
########### CamShift Tracking with Camera Input ############

################## By:- Khushhal Jindal ####################


##### Running the code: Tracking is implemented by selecting the object to be tracked by the mouse #####
############# To exit the window : Press Esc ###############

import cv2
import numpy as np

# A class named "CamShift" is defined which consists of variables and methods used to implement the CamShift algorithm for tracking objects:
class CamShift(): 

    # Function to initialize the variables and reading video frames: 
    def __init__(cls): 
        cls.FLAG_track = 0

        # Initializing Region of interest :
        cls.region_of_interest = None
        # Initializing coordinates of mouse click :
        cls.rect_coord = None

        # Creating a VideoCapture object in order to capture a video from camera:
        cls.cap = cv2.VideoCapture(0)
        
        # Initializing the current frame 'video_frame' of the video to process frame by frame :
        ret, cls.video_frame = cls.cap.read()

        # Creating a window with name 'Tracking' :
        cv2.namedWindow('Tracking', 1)

        # Setting mouse handler for the window 'Tracking' by calling the function 'MouseClick' :
        cv2.setMouseCallback('Tracking', cls.MouseClick, None)

    # The function below is used to select our region of interest in the video frame: 
    ## Function to define what happens when we click a mouse button :
    def MouseClick(cls, event, x, y, flags, param):
        
        # Event for which left mouse is being clicked, or released (giving click's x & y coordinates):
        if event == cv2.EVENT_LBUTTONDOWN:

            # Storing x and y coordiantes of mouse click to the points of region of interest to be used to track the object :
            cls.rect_coord = (x, y) 
        
            cls.FLAG_track = 0
            return

        if cls.rect_coord:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                # Getting the dimesions using shape of frame and storing it in variables s1 and s2:
                s1, s2 = cls.video_frame.shape[:2]

                # Taking the rectangle coordinates into variables a and b :
                a,b = cls.rect_coord

                # Setting the four co-ordinates of the rectangle :
                a1,b1 = np.maximum(0, np.minimum([a,b], [x, y]))
                a2,b2 = np.minimum([s2, s1], np.maximum([a,b], [x, y]))
                cls.region_of_interest = None
                if a2-a1 > 0 and b2-b1 > 0:
                    # Setting the region of interest window with coordinates:
                    cls.region_of_interest = (a1,b1,a2,b2) 
            else:
                cls.rect_coord = None 
                if cls.region_of_interest is not None: 
                    cls.FLAG_track = 1 

    def hue(cls):
        b = 24
        # Taking the histogram bins according to the shape of image's histogram: 
        bins = cls.histogram.shape[0]
        
        # Creating a histogram by initializing a 3d array for the given size with all array values initially equal to zero:
        img = np.zeros((256, bins*b, 3), np.uint8)
        
        for j in range(bins):

            # For all histogram bins, taking the integer values for selected channels :
            s1 = int(cls.histogram[j])

            # Choosing the region shape to be tracked representing mouse's click location:
            cv2.rectangle(img, (j*(b+2), 255), ((j+1)*(b-2), 255-s1), (int(180*(j/bins)), 255, 255), -1)
        
        # Coverting image from HSV color space to RGB color space in order to show the histogram image in RGB format:
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR) 


    def main(cls): 
        while True: # Loop used for processing all the frames

            # Taking each frame of the video:
            ret, cls.video_frame = cls.cap.read()

            # Copying the frame of the video ('new_w' is an image type):
            new_w = cls.video_frame.copy() 

            # Converting the tracked window (of region of interest) of the video to HSV color space :
            region_of_interest_hsv = cv2.cvtColor(cls.video_frame, cv2.COLOR_BGR2HSV)

            # Thresholding the HSV image to get the R,G,B colors :
            mask = cv2.inRange(region_of_interest_hsv, np.array((0., 30., 32.)), np.array((180., 255., 255.)))

            if cls.region_of_interest:
                a1,b1, a2,b2 = cls.region_of_interest
                
                # Estimated size of the window in between which object is tracked : 
                cls.window = (a1,b1, (a2-a1), (b2-b1))

                # Creating an HSV image with the given coordinates:
                image_hsv = region_of_interest_hsv[b1:b2, a1:a2]

                # Creating mask image in order to find histogram of any specific image region:  
                image_mask = mask[b1:b2, a1:a2]

                # Calculating the HSV histogram of arrays (with 16 values to represent histogram and range of HSV color space[0,180]) :
                histogram = cv2.calcHist( [image_hsv], [0], image_mask, [16], [0, 180] )

                # Normalizing the histogram with parameter as input and output histogram, gray scale vlaues range and 
                cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
                cls.histogram = histogram.reshape(-1)
                
                # Calling the 'hue' function to choose the region shape to be tracked based on histogram bins:
                cls.hue()
                # Setting the new region window for frame:
                region_new_w = new_w[b1:b2, a1:a2]

                # Bitwise NOT operation used to choose the shape of the object to be tracked as rectangle:
                cv2.bitwise_not(region_new_w, region_new_w)

            if cls.FLAG_track == 1:
                cls.region_of_interest = None

                # Calculate the Histogram Back Projection:
                ## Finding the corresponding histogram bin for different values collected from image channels :
                ### Calculating the probability values of each element with respect to the histogram's probability distribution :
                probability = cv2.calcBackProject([region_of_interest_hsv], [0], cls.histogram, [0, 180], 1)
                probability &= mask
                
                # Termination Condition for the algorithm specifying the minimum movement by 1 pixel along region of interest & a maximum of 5 iterations:
                terminate = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 1 )
                
                # CamSHift function giving us the new position of Region of interest and size, position of our desired object to be tracked:
                track, cls.window = cv2.CamShift(probability, cls.window, terminate) 

                # Setting the ellipse in order to define the shape of the tracked object by using the new position of region of interest :
                cv2.ellipse(new_w, track, (255, 0, 255), 3) 
                 
            # Showing the tracking window by drawing new position of the region of interest on the image:
            cv2.imshow('Tracking', new_w)

            ch = 0xFF & cv2.waitKey(5)
            if ch == 27:
                break

        # Closing any open windows:
        cv2.destroyAllWindows()

# Calling the 'main' function of the class using the class object:
CamShift().main() 