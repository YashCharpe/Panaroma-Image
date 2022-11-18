import os 
import sys
import streamlit as st
import pandas as pd
from io import BytesIO, StringIO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
 

STYLE = """
<style>
img{
    max-width:100%;
}
</style>

"""

def detectAndDescribe(image, method="sift"):
    """
    Compute key points and feature descriptors using an specific method
    """

    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()
        
    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)
    
    return (kps, features)

def createMatcher(method,crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)

def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)
        
    # Match descriptors.
    best_matches = bf.match(featuresA,featuresB)
    
    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key = lambda x:x.distance)
    return rawMatches

def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)

    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])
    
    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
        
        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
            reprojThresh)

        return (matches, H, status)
    else:
        return None

def merge_images(trainImg, queryImg, extractor='orb'):
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_RGB2GRAY)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_RGB2GRAY)
    
    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=extractor)
    
    # Feature matcher
    matches = matchKeyPointsBF(featuresA, featuresB, method=extractor)

    # Homography
    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
    
    if M is None:
        print('Error!')
        return
        
    (matches, H, status) = M
    
    # Apply Panaroma
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    
    # transform the panorama image to grayscale and threshold it 
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    c = max(cnts, key=cv2.contourArea)

    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    result = result[y:y + h, x:x + w]

    return result


def PanaromaImageCreation(images):

    result = images[0]
    for current in images[1:]:
        result = merge_images(current, result, 'brisk')

    return result


def main():
    st.header("Computer Vision Assignment 4 Panaroma Image")
    st.subheader("Name: Yash Charpe")
    st.subheader("Roll No: 181")
    #st.info(__doc__)
    st.markdown(STYLE,unsafe_allow_html=True)
    file1 = st.file_uploader("Upload PNG File",type=["png","jpg"],key=1)
    file2 = st.file_uploader("Upload PNG File",type=["png","jpg"],key=2)
    file3 = st.file_uploader("Upload PNG File",type=["png","jpg"],key=3)
    file4 = st.file_uploader("Upload PNG File",type=["png","jpg"],key=4)
    st.subheader("Original Image")
    show_file = st.empty()
    panaromaOutput = st.empty()
    
    if not file1:
        show_file.info("Please Upload a file: {}".format(' '.join(["png","jpg"])))
        return
    
    content1 = file1.getvalue()
    content2 = file2.getvalue()
    content3 = file3.getvalue()
    content4 = file4.getvalue()

    if isinstance(file1,BytesIO) and isinstance(file2,BytesIO) and isinstance(file3,BytesIO) and isinstance(file4,BytesIO):
        show_file.image(file1)
        show_file.image(file2)
        show_file.image(file3)
        show_file.image(file4)
        original_img1 = cv2.imdecode(np.frombuffer(content1,np.uint8),cv2.IMREAD_COLOR)
        original_img2 = cv2.imdecode(np.frombuffer(content2,np.uint8),cv2.IMREAD_COLOR)
        original_img3 = cv2.imdecode(np.frombuffer(content3,np.uint8),cv2.IMREAD_COLOR)
        original_img4 = cv2.imdecode(np.frombuffer(content4,np.uint8),cv2.IMREAD_COLOR)

        images = [original_img1,original_img2,original_img3,original_img4]

        panaromaImage = PanaromaImageCreation(images)
        panaromaOutput.image (panaromaImage)


    else:
        df = pd.read_csv(file1)
        st.dataframe(df.head(2))
    file1.close()
    file2.close()
    file3.close()
    file4.close()
       

main()
