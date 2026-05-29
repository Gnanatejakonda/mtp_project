import cv2
import numpy as np
import torch
import pathlib
import os
pathlib.PosixPath = pathlib.WindowsPath

# ── CONFIG ────────────────────────────────────────────────────────────────────
LEFT_VIDEO  = 'C:/drone videos/phantom02_left_half.mp4'
RIGHT_VIDEO = 'C:/drone videos/phantom02_right_half.mp4'
OUTPUT      = 'runs/detect/live_stitched_yolo.mp4'
WEIGHTS     = 'runs/train/exp8/weights/best.pt'
CONF_THRES  = 0.25
IOU_THRES   = 0.45
IMG_SIZE    = 960
DEVICE      = 'cpu'
LINE_THICKNESS = 2
# ─────────────────────────────────────────────────────────────────────────────


def load_yolo(weights, device):
    print(f"[YOLO] Loading model from {weights}...")
    model = torch.hub.load(
        '.',
        'custom',
        path=weights,
        source='local',
        force_reload=False
    )
    model.to(device)
    model.conf = CONF_THRES
    model.iou  = IOU_THRES
    model.eval()
    print(f"[YOLO] Model loaded. Classes: {model.names}")
    return model


def run_yolo(model, frame, img_size):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb, size=img_size)
    detections = results.xyxy[0].cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        label = f"{model.names[int(cls_id)]} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return frame, len(detections)


def compute_homography(frame_left, frame_right):
    gray_l = cv2.cvtColor(frame_left,  cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    kp_l, des_l = orb.detectAndCompute(gray_l, None)
    kp_r, des_r = orb.detectAndCompute(gray_r, None)

    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des_l, des_r), key=lambda x: x.distance)[:300]

    if len(matches) < 10:
        raise ValueError("Not enough matches for homography.")

    src_pts = np.float32([kp_l[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp_r[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    print(f"[Stitch] Homography inliers: {int(mask.sum())}")
    return H


def compute_canvas(H, h_l, w_l, h_r, w_r):
    corners_r   = np.float32([[0,0],[w_r,0],[w_r,h_r],[0,h_r]]).reshape(-1,1,2)
    corners_r_t = cv2.perspectiveTransform(corners_r, H)
    corners_l   = np.float32([[0,0],[w_l,0],[w_l,h_l],[0,h_l]]).reshape(-1,1,2)
    all_c       = np.concatenate([corners_l, corners_r_t], axis=0)

    x_min, y_min = np.int32(all_c.min(axis=0).ravel())
    x_max, y_max = np.int32(all_c.max(axis=0).ravel())

    return x_max-x_min, y_max-y_min, -x_min, -y_min


def precompute_masks(H_t, canvas_w, canvas_h, h_l, w_l, ox, oy, h_r, w_r):
    foot_r = cv2.warpPerspective(
        np.ones((h_r, w_r), dtype=np.float32), H_t, (canvas_w, canvas_h))
    foot_r = np.clip(foot_r, 0, 1)

    foot_l = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    y1s = max(0, oy);          y1e = min(canvas_h, oy + h_l)
    x1s = max(0, ox);          x1e = min(canvas_w, ox + w_l)
    foot_l[y1s:y1e, x1s:x1e] = 1.0

    overlap      = ((foot_l > 0.5) & (foot_r > 0.5)).astype(np.float32)
    alpha_l_2d   = foot_l.copy()
    alpha_r_2d   = foot_r.copy()
    overlap_cols = np.where(overlap.any(axis=0))[0]

    if len(overlap_cols) > 0:
        x_ol  = overlap_cols.min()
        x_or  = overlap_cols.max()
        width = max(x_or - x_ol, 1)
        xs    = np.arange(x_ol, x_or + 1)
        t     = (xs - x_ol) / width
        col_mask = overlap[:, x_ol:x_or+1]
        alpha_l_2d[:, x_ol:x_or+1] = np.where(col_mask, 1.0 - t, alpha_l_2d[:, x_ol:x_or+1])
        alpha_r_2d[:, x_ol:x_or+1] = np.where(col_mask, t,       alpha_r_2d[:, x_ol:x_or+1])

    return (alpha_l_2d[..., np.newaxis],
            alpha_r_2d[..., np.newaxis],
            y1s, y1e, x1s, x1e)


def stitch_frame(f_l, f_r, H_t, canvas_w, canvas_h,
                 alpha_l, alpha_r, y1s, y1e, x1s, x1e):
    warped_r = cv2.warpPerspective(f_r, H_t, (canvas_w, canvas_h),
                                   flags=cv2.INTER_LINEAR)
    canvas_l = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    canvas_l[y1s:y1e, x1s:x1e] = f_l[0:y1e-y1s, 0:x1e-x1s].astype(np.float32)

    blended = canvas_l * alpha_l + warped_r.astype(np.float32) * alpha_r
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    cap1 = cv2.VideoCapture(LEFT_VIDEO)
    cap2 = cv2.VideoCapture(RIGHT_VIDEO)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error: Could not open video files. Check your paths:")
        print(f"   LEFT  → {LEFT_VIDEO}")
        print(f"   RIGHT → {RIGHT_VIDEO}")
        return

    ret1, frame_l = cap1.read()
    ret2, frame_r = cap2.read()
    if not ret1 or not ret2:
        print("❌ Error: Could not read first frames.")
        return

    h_l, w_l = frame_l.shape[:2]
    h_r, w_r = frame_r.shape[:2]
    print(f"[Info] Left : {w_l}x{h_l}")
    print(f"[Info] Right: {w_r}x{h_r}")

    print("\n[1/4] Computing Homography...")
    H = compute_homography(frame_l, frame_r)

    print("[2/4] Computing Canvas Size...")
    canvas_w, canvas_h, ox, oy = compute_canvas(H, h_l, w_l, h_r, w_r)
    translation  = np.array([[1,0,ox],[0,1,oy],[0,0,1]], dtype=np.float64)
    H_translated = translation.dot(H)
    print(f"[Info] Canvas: {canvas_w}x{canvas_h}")

    print("[3/4] Pre-computing Blend Masks...")
    alpha_l, alpha_r, y1s, y1e, x1s, x1e = precompute_masks(
        H_translated, canvas_w, canvas_h, h_l, w_l, ox, oy, h_r, w_r)

    print("[4/4] Loading YOLO...")
    model = load_yolo(WEIGHTS, DEVICE)

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    fps    = cap1.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(OUTPUT, fourcc, fps, (canvas_w, canvas_h))

    print(f"\n🚀 Starting Live Pipeline...")
    print(f"   Canvas  : {canvas_w}x{canvas_h}")
    print(f"   FPS     : {fps}")
    print(f"   Output  : {OUTPUT}\n")

    frame_count = 0
    total_dets  = 0

    while True:
        if frame_count > 0:
            ret1, frame_l = cap1.read()
            ret2, frame_r = cap2.read()
            if not ret1 or not ret2:
                break

        # ── STEP A: Stitch ───────────────────────────────────────────────────
        stitched = stitch_frame(
            frame_l, frame_r, H_translated,
            canvas_w, canvas_h,
            alpha_l, alpha_r,
            y1s, y1e, x1s, x1e
        )

        # ── STEP B: YOLO ─────────────────────────────────────────────────────
        result_frame, num_dets = run_yolo(model, stitched, IMG_SIZE)
        total_dets += num_dets

        # ── STEP C: Save ─────────────────────────────────────────────────────
        out.write(result_frame)
        frame_count += 1

        if frame_count % 50 == 0:
            print(f"  Frame {frame_count} | Detections so far: {total_dets}")

    print(f"\n✅ Done!")
    print(f"   Frames processed : {frame_count}")
    print(f"   Total detections : {total_dets}")
    print(f"   Avg per frame    : {total_dets/max(frame_count,1):.2f}")
    print(f"   Output saved to  : {OUTPUT}")

    cap1.release()
    cap2.release()
    out.release()


if __name__ == '__main__':
    main()
