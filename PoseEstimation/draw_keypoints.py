# draw the keypoints
def draw_keypoints(frame, keypoints, confidence_threshold):
    # height, width, channels 
    h, w, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [h, w, 1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 8, (0,255,0), -1)
