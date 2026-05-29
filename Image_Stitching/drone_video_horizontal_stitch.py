import cv2
import numpy as np

# 1. Open the video files
# video1 = Left Camera, video2 = Right Camera
cap1 = cv2.VideoCapture('C:/drone videos/phantom02_left_half.mp4') 
cap2 = cv2.VideoCapture('C:/drone videos/phantom02_right_half.mp4')

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open video files. Check your file paths.")
else:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        print("Error: Could not read the first frames.")
    else:
        print("Calculating Homography from Frame 1...")

        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]

        # 3. Find Features and Matches using SIFT
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(frame1, None)
        kp2, des2 = sift.detectAndCompute(frame2, None)

        # Use FLANN matcher for better accuracy
        FLANN_INDEX_KDTREE = 1
        index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test to filter good matches
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        print(f"Good matches found: {len(good_matches)}")

        if len(good_matches) < 10:
            print("Not enough good matches. Try with videos that have more overlapping content.")
        else:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Find Homography: maps video2 (right) into video1 (left) coordinate space
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            print(f"Homography matrix:\n{H}")

            # 4. Calculate the correct canvas size by projecting corners of video2
            corners_v2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
            corners_v2_transformed = cv2.perspectiveTransform(corners_v2, H)

            # Combine corners of both videos to get the full bounding box
            corners_v1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
            all_corners = np.concatenate([corners_v1, corners_v2_transformed], axis=0)

            [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel())
            [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel())

            # Translation to shift everything into positive coordinates if needed
            translation = np.array([
                [1, 0, -x_min],
                [0, 1, -y_min],
                [0, 0, 1]
            ], dtype=np.float64)

            H_translated = translation.dot(H)

            canvas_width  = x_max - x_min
            canvas_height = y_max - y_min

            print(f"Canvas size: {canvas_width} x {canvas_height}")

            # 5. Prepare Video Writer
            fps    = cap1.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('stitched_output_l_r_claud.mp4', fourcc, fps, (canvas_width, canvas_height))

            print("Starting video processing...")

            frame_count = 0
            offset_x = -x_min
            offset_y = -y_min

            while True:
                if frame_count > 0:
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if not ret1 or not ret2:
                        break

                # --- Warp video2 onto the full canvas ---
                warped_v2 = cv2.warpPerspective(frame2, H_translated, (canvas_width, canvas_height))

                # --- Place video1 onto the canvas at the correct offset ---
                canvas = warped_v2.copy()

                # Region where video1 will be placed
                y1_start = offset_y
                y1_end   = offset_y + h1
                x1_start = offset_x
                x1_end   = offset_x + w1

                # Clamp to canvas bounds
                y1_start = max(0, y1_start)
                y1_end   = min(canvas_height, y1_end)
                x1_start = max(0, x1_start)
                x1_end   = min(canvas_width, x1_end)

                roi_h = y1_end - y1_start
                roi_w = x1_end - x1_start

                # --- Blend the overlapping region instead of hard paste ---
                v2_roi  = warped_v2[y1_start:y1_end, x1_start:x1_end]
                v1_crop = frame1[0:roi_h, 0:roi_w]

                # Create masks: where each frame has actual content (non-black)
                mask_v1 = (cv2.cvtColor(v1_crop, cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)
                mask_v2 = (cv2.cvtColor(v2_roi,  cv2.COLOR_BGR2GRAY) > 0).astype(np.float32)

                # Overlap zone: both have content → blend 50/50
                overlap = (mask_v1 * mask_v2)
                # Only v1 zone
                only_v1 = mask_v1 * (1 - mask_v2)
                # Only v2 zone
                only_v2 = mask_v2 * (1 - mask_v1)

                blended = np.zeros_like(v1_crop, dtype=np.float32)
                for c in range(3):
                    blended[:, :, c] = (
                        only_v1  * v1_crop[:, :, c] +
                        only_v2  * v2_roi[:, :, c]  +
                        overlap  * (0.5 * v1_crop[:, :, c] + 0.5 * v2_roi[:, :, c])
                    )

                canvas[y1_start:y1_end, x1_start:x1_end] = blended.astype(np.uint8)

                out.write(canvas)
                frame_count += 1

                if frame_count % 50 == 0:
                    print(f"Processed {frame_count} frames...")

            print(f"Finished! Successfully stitched {frame_count} frames.")
            print("Output saved as 'stitched_output_l_r.mp4'")

# 7. Clean up
cap1.release()
cap2.release()
if 'out' in locals():
    out.release()