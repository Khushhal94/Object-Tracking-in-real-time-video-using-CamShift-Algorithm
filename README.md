# CamShift Tracking of an object in real-time video

CamShift (Continuously Adaptive Meanshift) is a tracking algorithm based on meanshift algorithm that anticipates 
the movement of an object and quickly tracks it in the next frame of the video based on dynamic color probability distribution.
To track an object in a real-time video captured by our web camera, meanShift algorithm is applied 
to every single frame, and the initial window of each frame is just the output window of the prior frame. 
It adapts the tracking window size with the target object's rotation and size.
