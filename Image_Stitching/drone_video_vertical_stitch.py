import cv2
import numpy as np

# 1. Open the video files
# video1 = Bottom Camera, video2 = Top Camera
cap1 = cv2.VideoCapture('C:/drone videos/phantom02_bottom_half.mp4') 
cap2 = cv2.VideoCapture('C:/drone videos/phantom02_top_half.mp4')

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video files. Check your file paths.")
else:
    # 2. Read the VERY FIRST frame from both videos to calculate the math
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read the first frames.")
    else:
        print("Calculating Homography from Frame 1...")
        
        # 3. Find Features and Matches (ONLY ONCE)
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(frame1, None)
        kp2, des2 = sift.detectAndCompute(frame2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matching points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find Homography
        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        # 4. Calculate Canvas Size & Translation (Vertical Logic)
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        translation_matrix = np.array([
            [1, 0, 0],
            [0, 1, h2],
            [0, 0, 1]
        ], dtype=np.float32)

        H_translated = translation_matrix.dot(H)

        canvas_width = max(w1, w2)
        canvas_height = h1 + h2

        # 5. Prepare the Video Writer to save the output
        # Get the frames per second (FPS) from the first video to match the speed
        fps = cap1.get(cv2.CAP_PROP_FPS)
        
        # Define the codec and create VideoWriter object ('mp4v' is standard for .mp4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('C:/drone videos/stitched_output.mp4', fourcc, fps, (canvas_width, canvas_height))

        print("Starting video processing. This might take a minute depending on video length...")
        
        frame_count = 0
        
        # 6. Loop through the rest of the videos
        while True:
            # If it's the very first loop, we already have frame1 and frame2. 
            # Otherwise, read the next frames.
            if frame_count > 0:
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                # If either video ends, stop the loop
                if not ret1 or not ret2:
                    break

            # Warp the top frame using our PRE-CALCULATED matrix
            frame2_warped = cv2.warpPerspective(frame2, H_translated, (canvas_width, canvas_height))

            # Paste the bottom frame directly underneath
            frame2_warped[h2:h1+h2, 0:w1] = frame1

            # Write the stitched frame to the new video file
            out.write(frame2_warped)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Processed {frame_count} frames...")

        print(f"Finished! Successfully stitched {frame_count} frames.")
        print("You can now download 'stitched_output.mp4' from the Colab files panel.")

# 7. Clean up and close all files
cap1.release()
cap2.release()
if 'out' in locals():
    out.release()