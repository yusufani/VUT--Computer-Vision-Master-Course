# coding: utf-8
from __future__ import print_function

import numpy as np
import cv2

# This should help with the assignment:
# * Indexing numpy arrays http://scipy-cookbook.readthedocs.io/items/Indexing.html


def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', help='Input video file name.')
    parser.add_argument('-i', '--image', help='Input image file name.')
    args = parser.parse_args()
    return args


def image(imageFileName):
    # read image
    img = cv2.imread(imageFileName)## FILL
    if img is None:
        print("Error: Unable to read image file", imageFileName)
        exit(-1)
    
    # print image width, height, and channel count
    print("Image dimensions: ", img.shape )## FILL
    
    # Resize to width 400 and height 500 with bicubic interpolation.
    img = cv2.resize(img,(400,500))## FILL
    
    # Print mean image color and standard deviation of each color channel
    print('Image mean and standard deviation', cv2.meanStdDev(img) )  ## FILL
    
    # Fill horizontal rectangle with color 128.  
    # Position x1=50,y1=120 and size width=200, height=50
    ## FILL
    img_org = img.copy()
    cv2.rectangle(img, (50, 120), (250, 170), (128, 128, 128), -1)
    
    # write result to file
    cv2.imwrite('rectangle.png', img)
    
    # Fill every third column in the top half of the image black.
    # The first column sould be black.  
    # The rectangle should not be visible.
    img = img_org # For rectangle
    selected_rows = np.arange(0,len(img)//2,1)
    selected_columns = np.arange(0,len(img[0]),3)

    img[selected_rows[:, None], selected_columns] = [0,0,0]



    ## FILL
    
    # write result to file
    cv2.imwrite('striped.png', img)
    
    # Set all pixels with any a value of any collor channel lower than 100 to black (0,0,0).
    ## FILL
    black = np.where(((img[:,:,0]<100) |  (img[:,:,1]<100) | (img[:,:,2]<100)))
    img[black] = (0,0,0)


    #write result to file
    cv2.imwrite('clip.png', img)
   
    
def video(videoFileName):

    # open video file and get basic information
    videoCapture = cv2.VideoCapture(videoFileName)
    frameRate =  videoCapture.get(cv2.CAP_PROP_FPS)
    frame_width = int ( videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) )
    frame_height= int (videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) )
    if not videoCapture.isOpened():
        print("Error: Unable to open video file for reading", videoFileName)
        exit(-1)

    # open video file for writing
    videoWriter  = cv2.VideoWriter(
        'videoOut.avi', cv2.VideoWriter_fourcc('M','J','P','G'),
        frameRate, (frame_width, frame_height))
    if not videoWriter.isOpened():
        print("Error: Unable to open video file for writing", videoFileName)
        exit(-1)

    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        if not ret:
            break

        # Flip image upside down.
        cv2.flip(frame,1)

        # Add white noise (additive noise with normal distribution).
        # Standard deviation should be 5.
        # use np.random
        normal_dist = np.random.normal(0, 5, (frame_height,frame_width,3 )).astype("uint8")
        frame += normal_dist

        ## FILL

        # Add gamma correction.
        # y = x^1.2 -- the image to the power of 1.2
        ## FILL
        for i in range(256):
            frame[0, i] = np.clip(pow(i / 255.0, 1.2) * 255.0, 0, 255)


        # Dim blue color to half intensity.
        ## FILL
        frame[:,:,0] //= 2

        # Invert colors.
        ## FILL
        frame = 255 - frame


        # Display the processed frame.
        cv2.imshow("Output", frame)
        # Write the resulting frame to the video file.
        videoWriter.write(frame)

        # End the processing on pressing Escape.
        if cv2.waitKey(30) == 27 :
            break
        ## FILL
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    videoCapture.release()
    videoWriter.release()


def main():
    args = parseArguments()
    np.random.seed(1)
    image(args.image)
    video(args.video)

if __name__ == "__main__":
    main()

