"""
annotate_video.py  - New dataset version (V.SRT + position.csv, no correlated CSV)
====================================================================================
Handles the new dataset where:
  - V.SRT: observer drone position + focal length + gb_yaw (camera azimuth)
  - position.csv: evader (target) drone position (ArduPilot log)
  - NO correlated CSV - we compute correlation ourselves using clock offset
  - Clock offset: position.csv_time = V.SRT_time + 1.348s
    (evader takeoff at V.SRT frame 1616 matches position.csv row 501)

KEY FINDINGS from data analysis:
  - Focal length switches from 24mm (frames 1-3990) to 161mm (frames 3991+)
  - At 24mm: HFOV=39.6 deg, evader enters frame ~frame 3200
  - At 161mm: HFOV=6.15 deg (very narrow), evader is mostly outside FOV
  - Evader is at same lat/lon as observer in early frames (vertical takeoff)`   ``
    then moves away horizontally
  - Camera gb_yaw=26 deg (NNE), evader initially due North -> offset causes out-of-frame

USAGE:
  python annotate_video.py --video DJI_20260420174320_0001.MP4 \
    --vsrt DJI_20260420174320_0001_V.SRT \
    --pos  position.csv \
    --sensor 4/3 \
    --clock-offset 1.348 \
    --start 3000 --end 4200 \
    --output test_clip.mp4

  # Full video:
  python annotate_video.py --video DJI_20260420174320_0001.MP4 \
    --vsrt DJI_20260420174320_0001_V.SRT \
    --pos  position.csv \
    --sensor 4/3 \
    --output annotated_video.mp4
"""

import argparse, math, re, sys, time
import cv2
import numpy as np
import pandas as pd
from collections import deque

SENSORS = {
    '4/3':   (17.3, 13.0),
    '1inch': (13.2,  8.8),
    '1/2':   ( 6.4,  4.8),
    '1/1.7': ( 7.6,  5.7),
}

# ─────────────────────────────────────────────────────────────────────────────
#  ALTITUDE NOISE FILTER
# ─────────────────────────────────────────────────────────────────────────────

class AltitudeFilter:
    """Simple moving average filter to smooth altitude readings and reduce noise."""
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
    
    def filter(self, alt_value):
        """Apply moving average filter to altitude."""
        if alt_value is None:
            return None
        self.buffer.append(alt_value)
        return np.mean(list(self.buffer))

# ─────────────────────────────────────────────────────────────────────────────
#  PURE MATH  (same projection as before)
# ─────────────────────────────────────────────────────────────────────────────

def compute_fov(focal_mm, sw, sh):
    return (math.degrees(2*math.atan(sw/(2*focal_mm))),
            math.degrees(2*math.atan(sh/(2*focal_mm))))

def haversine_bearing(lat1, lon1, lat2, lon2):
    lat1r,lon1r,lat2r,lon2r = map(math.radians,[lat1,lon1,lat2,lon2])
    dlon = lon2r-lon1r
    x = math.sin(dlon)*math.cos(lat2r)
    y = math.cos(lat1r)*math.sin(lat2r)-math.sin(lat1r)*math.cos(lat2r)*math.cos(dlon)
    return (math.degrees(math.atan2(x,y))+360)%360

def haversine_dist(lat1,lon1,lat2,lon2):
    R=6371000.0
    lat1r,lat2r=math.radians(lat1),math.radians(lat2)
    dlat=lat2r-lat1r; dlon=math.radians(lon2-lon1)
    a=math.sin(dlat/2)**2+math.cos(lat1r)*math.cos(lat2r)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(a))

def compute_elevation(tgt_alt, obs_alt, horiz_dist):
    # positive = target ABOVE camera
    return math.degrees(math.atan2(tgt_alt-obs_alt, horiz_dist))

def project_3d_to_pixel(obs_lat,obs_lon,obs_alt,
                         tgt_lat,tgt_lon,tgt_alt,
                         cam_yaw_deg, cam_pitch_deg,
                         focal_mm, sw, sh, W, H):
    R=6371000.0
    lat1r=math.radians(obs_lat); lon1r=math.radians(obs_lon)
    lat2r=math.radians(tgt_lat); lon2r=math.radians(tgt_lon)
    dlon=lon2r-lon1r

    N=(lat2r-lat1r)*R
    E=dlon*math.cos(lat1r)*R
    D=-(tgt_alt-obs_alt)

    yaw=math.radians(cam_yaw_deg); pitch=math.radians(cam_pitch_deg)
    fN=math.cos(yaw)*math.cos(pitch); fE=math.sin(yaw)*math.cos(pitch); fD=-math.sin(pitch)
    rN=-math.sin(yaw); rE=math.cos(yaw); rD=0.0
    dN_ax=fE*rD-fD*rE; dE_ax=fD*rN-fN*rD; dD_ax=fN*rE-fE*rN

    Zc=N*fN+E*fE+D*fD
    Xc=N*rN+E*rE+D*rD
    Yc=N*dN_ax+E*dE_ax+D*dD_ax

    dbg={'N':N,'E':E,'D':D,'Xc':Xc,'Yc':Yc,'Zc':Zc}

    if Zc<=0:
        return None,None,dbg

    fx=(focal_mm/sw)*W; fy=(focal_mm/sh)*H
    px=fx*(Xc/Zc)+W/2
    py=fy*(Yc/Zc)+H/2
    dbg['fx']=fx; dbg['fy']=fy; dbg['px_raw']=px; dbg['py_raw']=py
    return int(round(px)), int(round(py)), dbg


# ─────────────────────────────────────────────────────────────────────────────
#  SRT PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_vsrt(path):
    """
    Parse V.SRT -> {frame: {ts_ist, focal, obs_lat, obs_lon, obs_alt, gb_yaw, gb_pitch}}
    ts_ist is the IST timestamp string from the SRT file.
    
    NOTE: gb_yaw is aircraft BODY heading, not gimbal yaw!
    We use GPS bearing (from lat/lon) as the camera yaw instead.
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        text = f.read()
    
    # Check if gimbal_yaw exists in the file
    has_gimbal_yaw = 'gimbal_yaw' in text.lower()
    
    pat = re.compile(
        r'FrameCnt:\s*(\d+).*?'
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\s*\n'
        r'.*?focal_len:\s*([\d.]+).*?'
        r'latitude:\s*([\d.-]+).*?longitude:\s*([\d.-]+).*?'
        r'rel_alt:\s*([\d.-]+)\s+abs_alt:\s*([\-\d.]+).*?'
        r'gb_yaw:\s*([\-\d.]+)\s+gb_pitch:\s*([\-\d.]+)\s+gb_roll:\s*([\-\d.]+)',
        re.DOTALL)
    result = {}
    for m in pat.finditer(text):
        fn = int(m.group(1))
        result[fn] = {
            'ts_ist':   m.group(2),
            'focal':    float(m.group(3)),
            'obs_lat':  float(m.group(4)),
            'obs_lon':  float(m.group(5)),
            'obs_alt':  float(m.group(6)),   # rel_alt (relative to observer home)
            'gb_yaw':   float(m.group(8)),   # aircraft body yaw (NOT camera azimuth)
            'gb_pitch': float(m.group(9)),   # gimbal pitch (used as camera pitch)
            'gb_roll':  float(m.group(10)),
        }
    
    if has_gimbal_yaw:
        print("[INFO] gimbal_yaw field detected in V.SRT - consider updating regex to extract it")
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CORRELATED CSV LOOKUP (frame-accurate synced data)
# ─────────────────────────────────────────────────────────────────────────────

def build_correlated_lookup(corr_csv_path):
    """
    Load correlated CSV with frame-accurate synchronized data.
    
    Uses UTC timestamp as index instead of frame number to handle multiple
    entries per frame (e.g., multiple cameras).
    
    Expected columns:
    - frame: frame number
    - utc_ts: UTC timestamp (ISO format)
    - primary_lat/lon/alt: observer drone (from V.SRT)
    - secondary_lat/lon/alt: target drone (being tracked)
    
    Returns DataFrame sorted by UTC timestamp.
    """
    df = pd.read_csv(corr_csv_path)
    
    if 'utc_ts' not in df.columns:
        raise ValueError("Correlated CSV must have 'utc_ts' column")
    
    # Convert to datetime - use format='mixed' to handle varying formats
    df['utc_ts'] = pd.to_datetime(df['utc_ts'], format='ISO8601')
    df = df.sort_values('utc_ts')
    
    print(f"  Loaded {len(df)} rows from correlated CSV")
    print(f"  Time range: {df['utc_ts'].min()} to {df['utc_ts'].max()}")
    
    return df

def build_position_lookup(pos_csv_path):
    """[DEPRECATED] Load position.csv - use build_correlated_lookup() instead."""
    df = pd.read_csv(pos_csv_path)
    df['GPS_UTC'] = pd.to_datetime(df['GPS_UTC'], utc=True, format='mixed')
    df = df.sort_values('GPS_UTC').reset_index(drop=True)
    return df

def get_evader_position_from_correlated(df_corr, utc_ts, camera_type='RGB'):
    """
    Get target drone position AND camera orientation from correlated CSV using UTC timestamp.
    
    Args:
        df_corr: DataFrame with UTC timestamp sorted (from build_correlated_lookup)
        utc_ts: UTC timestamp (pandas Timestamp with UTC tz)
        camera_type: Filter by camera type ('RGB' or 'IR' or None for first match)
    
    Returns: {tgt_lat, tgt_lon, tgt_alt, cam_yaw, cam_pitch, lag_s} or None
    
    KEY: camera yaw/pitch are in GLOBAL FRAME (world coordinates)
    Uses UTC timestamp for robust lookup even with multiple entries per frame.
    """
    try:
        # Find rows within 100ms of the target timestamp
        time_diffs = (df_corr['utc_ts'] - utc_ts).abs()
        candidates = df_corr[time_diffs <= pd.Timedelta(milliseconds=100)]
        
        if len(candidates) == 0:
            return None
        
        # Filter by camera type if specified
        if camera_type and 'camera' in candidates.columns:
            filtered = candidates[candidates['camera'] == camera_type]
            if len(filtered) > 0:
                candidates = filtered
        
        # Use the one closest in time
        best_idx = time_diffs[candidates.index].idxmin()
        row = df_corr.loc[best_idx]
        best_dt = abs((row['utc_ts'] - utc_ts).total_seconds())
        
        return {
            'tgt_lat':  float(row['secondary_lat']),
            'tgt_lon':  float(row['secondary_lon']),
            'tgt_alt':  float(row['secondary_rel_home_alt_m']),
            'cam_yaw':  float(row['primary_gb_yaw']),      # ← GLOBAL frame camera yaw
            'cam_pitch': float(row['primary_gb_pitch']),   # ← GLOBAL frame camera pitch
            'lag_s':    best_dt,  # Time offset from exact timestamp match
            'camera':   str(row.get('camera', 'UNKNOWN')),
            'horiz_dist': float(row.get('horizontal_dist_m', 0)),
            'vert_diff':  float(row.get('vertical_diff_m', 0)),
        }
    except Exception as e:
        return None

def get_evader_position(df_pos, ts_ist_str, clock_offset_s):
    """
    [DEPRECATED] Get evader position from position.csv using clock offset.
    
    Use get_evader_position_from_correlated() instead for better accuracy.
    """
    try:
        ts_ist = pd.Timestamp(ts_ist_str, tz='Asia/Kolkata')
        ts_utc = ts_ist.tz_convert('UTC')
        target_utc = ts_utc + pd.Timedelta(seconds=clock_offset_s)

        # Binary search for nearest timestamp
        idx = df_pos['GPS_UTC'].searchsorted(target_utc)
        idx = max(0, min(len(df_pos)-1, idx))

        # Check neighbours
        best_idx = idx
        best_dt = abs((df_pos.iloc[idx]['GPS_UTC'] - target_utc).total_seconds())
        for ni in [idx-1, idx+1]:
            if 0 <= ni < len(df_pos):
                dt = abs((df_pos.iloc[ni]['GPS_UTC'] - target_utc).total_seconds())
                if dt < best_dt:
                    best_dt = dt; best_idx = ni

        row = df_pos.iloc[best_idx]
        return {
            'tgt_lat':  float(row['POS_Lat']),
            'tgt_lon':  float(row['POS_Lng']),
            'tgt_alt':  float(row['POS_RelHomeAlt_m']),
            'lag_s':    best_dt,
        }
    except Exception as e:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  ANNOTATION
# ─────────────────────────────────────────────────────────────────────────────

# Smoothing state for reducing jitter at distance
_cam_angle_state = {'prev_yaw': None, 'prev_pitch': None}

def annotate_frame(img, frame_num, srt_f, evader, hfov, vfov, sw, sh, box_deg,
                   cam_yaw_offset=0, cam_pitch_offset=0, 
                   yaw_distance_scale_24mm=0, yaw_distance_scale_161mm=0,
                   pitch_offset_24mm=0, pitch_offset_106mm=0, pitch_offset_161mm=0,
                   pitch_distance_scale_24mm=0, pitch_distance_scale_161mm=0,
                   distance_ref=100.0,
                   auto_pitch=False, debug=False, use_global_angles=False, smooth_alpha=0.3):
    H, W = img.shape[:2]
    out  = img.copy()

    obs_lat = srt_f['obs_lat'];  obs_lon = srt_f['obs_lon'];  obs_alt = srt_f['obs_alt']
    tgt_lat = evader['tgt_lat']; tgt_lon = evader['tgt_lon']; tgt_alt = evader['tgt_alt']
    focal_mm= srt_f['focal']
    lag_s   = evader['lag_s']
    gb_yaw  = srt_f['gb_yaw']    # Always available from V.SRT
    gb_pitch= srt_f['gb_pitch']  # Always available from V.SRT

    # All quantities from pure math
    bearing   = haversine_bearing(obs_lat, obs_lon, tgt_lat, tgt_lon)
    horiz     = haversine_dist(obs_lat, obs_lon, tgt_lat, tgt_lon)
    alt_diff  = tgt_alt - obs_alt
    elevation = compute_elevation(tgt_alt, obs_alt, horiz)
    dist_3d   = math.sqrt(horiz**2 + alt_diff**2)

    # Camera orientation: Use GLOBAL FRAME angles from correlated CSV
    # CSV yaw is reliable for tracking; apply distance-dependent offset scaling
    if use_global_angles and 'cam_yaw' in evader and 'cam_pitch' in evader:
        # ✓ Use measured camera angles from CSV (gimbal orientation)
        # Apply distance-dependent offset: closer drones need more correction, farther drones need less
        # This counteracts gimbal calibration errors that worsen at distance
        # Formula: effective_offset = base_offset * (ref_distance / (current_distance * 20))
        
        # Distance-dependent offset scaling for yaw (20x denominator for gentler scaling)
        if horiz > 0:
            distance_scale_factor = distance_ref / (horiz * 50)  # Inverse distance weighting with 20x denominator
        else:
            distance_scale_factor = 1.0
        
        effective_yaw_offset = cam_yaw_offset * distance_scale_factor
        
        cam_yaw = evader['cam_yaw'] + effective_yaw_offset
        
        # Apply focal-length dependent pitch offset WITH distance correction
        if auto_pitch:
            cam_pitch = elevation
        else:
            # Keep using CSV pitch (more reliable than computed elevation)
            csv_pitch_base = evader.get('cam_pitch', elevation)
            
            # Dynamic interpolation based on actual focal length (24-161mm)
            if focal_mm <= 24:
                focal_offset = pitch_offset_24mm
                dist_scale = pitch_distance_scale_24mm
            elif focal_mm <= 106:
                # Interpolate focal offset between 24mm and 106mm
                t = (focal_mm - 24.0) / (106.0 - 24.0)
                focal_offset = pitch_offset_24mm + t * (pitch_offset_106mm - pitch_offset_24mm)
                # Interpolate distance scale
                dist_scale = pitch_distance_scale_24mm + t * (pitch_distance_scale_161mm - pitch_distance_scale_24mm)
            elif focal_mm <= 161:
                # Interpolate focal offset between 106mm and 161mm
                t = (focal_mm - 106.0) / (161.0 - 106.0)
                focal_offset = pitch_offset_106mm + t * (pitch_offset_161mm - pitch_offset_106mm)
                # Interpolate distance scale
                dist_scale = pitch_distance_scale_24mm + (1 - t) * (pitch_distance_scale_161mm - pitch_distance_scale_24mm)
            else:
                # Beyond 161mm, use 161mm offset
                focal_offset = pitch_offset_161mm
                dist_scale = pitch_distance_scale_161mm
            
            # DISTANCE-DEPENDENT CORRECTION (physics-based parallax/triangulation)
            # Offset changes with distance: offset = base + scale × (distance - ref_distance)
            distance_correction = dist_scale * (horiz - distance_ref) if horiz > 0 else 0
            
            # Use CSV pitch as base (more reliable), apply offsets
            cam_pitch = csv_pitch_base + cam_pitch_offset + focal_offset + distance_correction
    else:
        # ✗ Fallback: compute from bearing
        cam_yaw = bearing + cam_yaw_offset
        if auto_pitch:
            cam_pitch = elevation
        else:
            cam_pitch = gb_pitch + cam_pitch_offset
    
    # SMOOTHING: Apply exponential moving average to reduce jitter at distance
    # smooth_value = alpha * new + (1 - alpha) * previous
    # Higher alpha = more responsive, lower = more smooth (less jitter)
    if _cam_angle_state['prev_yaw'] is not None:
        cam_yaw = smooth_alpha * cam_yaw + (1 - smooth_alpha) * _cam_angle_state['prev_yaw']
    if _cam_angle_state['prev_pitch'] is not None:
        cam_pitch = smooth_alpha * cam_pitch + (1 - smooth_alpha) * _cam_angle_state['prev_pitch']
    
    # Update state for next frame
    _cam_angle_state['prev_yaw'] = cam_yaw
    _cam_angle_state['prev_pitch'] = cam_pitch
    
    if debug and frame_num % 30 == 0:
        print(f"\n[Frame {frame_num}] Global Frame Camera Angles:")
        if use_global_angles:
            print(f"  [OK] Using GLOBAL FRAME angles | yaw={evader['cam_yaw']:.2f}deg, pitch={evader['cam_pitch']:.2f}deg")
        else:
            print(f"  [NO] Fallback: Computed from bearing={bearing:.2f}deg")

    # 3D -> 2D
    px, py, dbg = project_3d_to_pixel(
        obs_lat, obs_lon, obs_alt,
        tgt_lat, tgt_lon, tgt_alt,
        cam_yaw, cam_pitch,
        focal_mm, sw, sh, W, H
    )

    bw = max(40, int(box_deg * W / hfov)) * 2
    bh = max(30, int(box_deg * H / vfov)) * 2

    # DISTANCE-DEPENDENT FOV EXPANSION
    # At far distances, the effective FOV boundary extends because:
    # - Angular measurement errors scale with distance
    # - Gimbal calibration offsets become less significant
    # - The drone should be visible if it's within the ACTUAL camera FOV
    fov_expansion = 1.0 + max(0, (horiz - 50.0) / 200.0)  # Expand by ~0.5% per meter beyond 50m
    expanded_hfov = hfov * fov_expansion
    expanded_vfov = vfov * fov_expansion
    
    # Check if target is within expanded FOV cone
    # Using projected pixel position as proxy for angular position
    within_h_fov = px is not None and (0 <= px < W)
    within_v_fov = py is not None and (0 <= py < H)
    
    # Determine status and colour
    if px is None:
        status = "BEHIND_CAM"; color = (0, 0, 255)
        px, py = W//2, H//2
    elif within_h_fov and within_v_fov:
        status = "IN_FRAME"; color = (0, 255, 0)
    else:
        # OUT_OF_FRAME: target is outside camera FOV or behind
        status = "OUT_OF_FRAME"; color = (0, 140, 255)
        # Clamp to frame for visualization
        px = max(bw//2, min(W-1-bw//2, px)) if px is not None else W//2
        py = max(bh//2, min(H-1-bh//2, py)) if py is not None else H//2

    if debug and frame_num % 30 == 0:
        N_=dbg.get('N',0);  E_=dbg.get('E',0);  D_=dbg.get('D',0)
        Xc=dbg.get('Xc',0); Yc=dbg.get('Yc',0); Zc=dbg.get('Zc',0)
        fx=dbg.get('fx',0); fy=dbg.get('fy',0)
        px_r=dbg.get('px_raw',px); py_r=dbg.get('py_raw',py)
        csv_yaw_base = evader.get('cam_yaw', 0) if use_global_angles else 0
        bearing_offset = bearing - csv_yaw_base
        if bearing_offset > 180:
            bearing_offset -= 360
        elif bearing_offset < -180:
            bearing_offset += 360
        
        print(f"  [f{frame_num}] Camera: focal={focal_mm}mm  sensor={sw}x{sh}mm  video={W}x{H}")
        print(f"  [f{frame_num}] GPS bearing={bearing:.2f}deg  CSV yaw={csv_yaw_base:.2f}deg  dynamic_offset={bearing_offset:.2f}deg  final_yaw={cam_yaw:.2f}deg")
        print(f"  [f{frame_num}] NED: N={N_:.1f}m  E={E_:.1f}m  D={D_:.1f}m  distance={horiz:.1f}m")
        print(f"  [f{frame_num}] CamFrame: Xc={Xc:.3f}m  Yc={Yc:.3f}m  Zc={Zc:.3f}m")
        print(f"  [f{frame_num}] Pixels: px_raw={px_r:.1f}  py_raw={py_r:.1f}  |  FINAL: px={px}  py={py}  {status}")


    # Draw box
    x1=max(0,px-bw//2); y1=max(0,py-bh//2)
    x2=min(W-1,px+bw//2); y2=min(H-1,py+bh//2)
    cv2.rectangle(out,(x1,y1),(x2,y2),color,3)
    tk=18
    for (qx,qy) in [(x1,y1),(x2,y1),(x1,y2),(x2,y2)]:
        ddx=1 if qx==x1 else -1; ddy=1 if qy==y1 else -1
        cv2.line(out,(qx,qy),(qx+ddx*tk,qy),color,4)
        cv2.line(out,(qx,qy),(qx,qy+ddy*tk),color,4)

    cv2.putText(out, f"DRONE  {dist_3d:.1f}m  [{status}]",
                (x1, max(y1-14, 24)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    # Compass rose
    ar=55; acx=W-90; acy=90
    cv2.circle(out,(acx,acy),ar,(0,0,0),-1)
    cv2.circle(out,(acx,acy),ar,(0,255,180),2)
    cv2.line(out,(acx,acy-ar+5),(acx,acy-ar+16),(0,255,180),2)
    cv2.putText(out,"N",(acx-7,acy-ar+33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,180),1)
    a=math.radians(bearing-90)
    cv2.arrowedLine(out,(acx,acy),(int(acx+(ar-13)*math.cos(a)),int(acy+(ar-13)*math.sin(a))),
                    (0,255,0),2,tipLength=0.3)
    cv2.putText(out,f"{bearing:.1f}d",(acx-25,acy+11),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,255,0),1)

    # HUD panel — 6 lines, all raw values
    Xc=dbg.get('Xc',0); Yc=dbg.get('Yc',0); Zc=dbg.get('Zc',0)
    N_=dbg.get('N',0);  E_=dbg.get('E',0);  D_=dbg.get('D',0)
    px_r=dbg.get('px_raw',px); py_r=dbg.get('py_raw',py)

    hud = [
        # Line 1: frame, focal, FOV, status
        (f"Frame:{frame_num}  focal:{focal_mm}mm  sensor:{sw}x{sh}mm  "
         f"HFOV:{hfov:.3f}deg  VFOV:{vfov:.3f}deg  "
         f"status:[{status}]  pos_lag:{lag_s:.3f}s"),

        # Line 2: observer (from V.SRT)
        (f"[V.SRT] obs_lat:{obs_lat:.8f}  obs_lon:{obs_lon:.8f}  "
         f"obs_alt:{obs_alt:.4f}m  gb_yaw:{gb_yaw:.2f}deg  "
         f"gb_pitch:{gb_pitch:.2f}deg"),

        # Line 3: target (from position.csv)
        (f"[pos.csv] tgt_lat:{tgt_lat:.8f}  tgt_lon:{tgt_lon:.8f}  "
         f"tgt_alt:{tgt_alt:.4f}m  "
         f"horiz:{horiz:.3f}m  dist_3d:{dist_3d:.3f}m"),

        # Line 4: angles
        (f"bearing(GPS):{bearing:.4f}deg  elevation(GPS):{elevation:.4f}deg  "
         f"alt_diff(tgt-obs):{alt_diff:+.4f}m  "
         f"cam_yaw:{cam_yaw:.2f}deg  cam_pitch:{cam_pitch:.2f}deg  "
         f"[GLOBAL FRAME]"),

        # Line 5: 3D camera coords
        (f"NED: N={N_:.3f}m  E={E_:.3f}m  D={D_:.3f}m  |  "
         f"CamFrame: Xc={Xc:.3f}m  Yc={Yc:.3f}m  Zc={Zc:.3f}m"),

        # Line 6: 2D result
        (f"px_raw:{px_r:.2f}  py_raw:{py_r:.2f}  |  "
         f"pixel:({px},{py})  offset_from_centre:({px-W//2:+d},{py-H//2:+d})px  "
         f"box:{bw}x{bh}px  in_frame:{status=='IN_FRAME'}"),
    ]

    lh=26; ph=len(hud)*lh+14
    bg=out.copy()
    cv2.rectangle(bg,(0,H-ph),(W,H),(0,0,0),-1)
    cv2.addWeighted(bg,0.65,out,0.35,0,out)
    for i,line in enumerate(hud):
        cv2.putText(out,line,(10,H-ph+10+i*lh),
                    cv2.FONT_HERSHEY_SIMPLEX,0.52,(0,255,180),1,cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",         required=True)
    ap.add_argument("--vsrt",          required=True,
                    help="V.SRT file (observer drone telemetry)")
    ap.add_argument("--correlated",    default=None,
                    help="Correlated CSV with frame-accurate synced data (RECOMMENDED)")
    ap.add_argument("--pos",           default=None,
                    help="[DEPRECATED] position.csv - use --correlated instead")
    ap.add_argument("--sensor",        default="4/3",
                    choices=list(SENSORS.keys())+["custom"])
    ap.add_argument("--sw",            type=float, default=None)
    ap.add_argument("--sh",            type=float, default=None)
    ap.add_argument("--box-size",      type=float, default=3.0,
                    help="Bounding box size in degrees (default: 3.0 - increased for better visibility)")
    ap.add_argument("--clock-offset",  type=float, default=1.348,
                    help="[DEPRECATED] Seconds to add to V.SRT UTC time - use --correlated instead")
    ap.add_argument("--cam-yaw-offset",   type=float, default=0,
                    help="Gimbal yaw calibration offset in degrees (add to all frames)")
    ap.add_argument("--yaw-distance-scale-24mm", type=float, default=0,
                    help="Distance-dependent yaw scale factor for 24mm (deg/meter)")
    ap.add_argument("--yaw-distance-scale-161mm", type=float, default=0,
                    help="Distance-dependent yaw scale factor for 161mm (deg/meter)")
    ap.add_argument("--cam-pitch-offset", type=float, default=0,
                    help="Gimbal pitch calibration offset in degrees")
    ap.add_argument("--pitch-offset-24mm", type=float, default=0,
                    help="Pitch offset calibration for 24mm focal length")
    ap.add_argument("--pitch-offset-106mm", type=float, default=0,
                    help="Pitch offset calibration for 106mm focal length")
    ap.add_argument("--pitch-offset-161mm", type=float, default=0,
                    help="Pitch offset calibration for 161mm focal length")
    ap.add_argument("--pitch-distance-scale-24mm", type=float, default=0,
                    help="Distance-dependent pitch scale factor for 24mm (deg/meter)")
    ap.add_argument("--pitch-distance-scale-161mm", type=float, default=0,
                    help="Distance-dependent pitch scale factor for 161mm (deg/meter)")
    ap.add_argument("--distance-ref", type=float, default=100.0,
                    help="Reference distance for dynamic offset calibration (default: 100m)")
    ap.add_argument("--smooth-alpha", type=float, default=0.3,
                    help="Smoothing factor (0-1): 0=max smooth/no jitter, 1=no smooth/responsive (default: 0.3)")
    ap.add_argument("--auto-pitch",    action="store_true",
                    help="Auto-calculate pitch from elevation instead of using gb_pitch")
    ap.add_argument("--debug",         action="store_true",
                    help="Print detailed debugging info for every frame")
    ap.add_argument("--start",         type=int, default=1)
    ap.add_argument("--end",           type=int, default=None)
    ap.add_argument("--output",        default="annotated_video.mp4")
    args = ap.parse_args()

    if args.sensor=="custom":
        if not args.sw or not args.sh: sys.exit("[ERROR] custom needs --sw --sh")
        sw,sh=args.sw,args.sh
    else:
        sw,sh=SENSORS[args.sensor]

    # ── Load data ─────────────────────────────────────────────────────────
    print("Parsing V.SRT...", end=" ", flush=True)
    srt_data = parse_vsrt(args.vsrt)
    print(f"OK  ({len(srt_data)} frames, range {min(srt_data)}-{max(srt_data)})")

    # Report focal length breakdown
    from collections import Counter
    focal_counts = Counter(v['focal'] for v in srt_data.values())
    for f,c in sorted(focal_counts.items()):
        hfov_f = math.degrees(2*math.atan(sw/(2*f)))
        print(f"  Focal {f}mm: {c} frames  HFOV={hfov_f:.2f}deg")

    print("Loading target drone data...", end=" ", flush=True)
    if args.correlated:
        # Use correlated CSV (RECOMMENDED - frame-accurate)
        df_corr = build_correlated_lookup(args.correlated)
        print(f"OK  (correlated.csv)")
        print(f"  Frame range in CSV: {min(df_corr.index)} – {max(df_corr.index)}")
        use_correlated = True
    elif args.pos:
        # Fallback to position.csv (DEPRECATED - clock offset based)
        df_pos = build_position_lookup(args.pos)
        print(f"OK  ({len(df_pos)} rows)")
        print(f"  Clock offset: {args.clock_offset:+.3f}s  (using legacy position.csv)")
        use_correlated = False
    else:
        print("[ERROR] Must provide either --correlated or --pos")
        sys.exit(1)
    print(f"Camera calibration:")
    print(f"  Yaw offset: {args.cam_yaw_offset:+.2f}deg  |  "
          f"Pitch offset: {args.cam_pitch_offset:+.2f}deg  |  "
          f"Auto-pitch: {args.auto_pitch}")
    print()

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit(f"[ERROR] Cannot open: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {W}x{H}  {fps:.3f}fps  {total} frames  ({total/fps/60:.2f} min)")

    start_frame = max(1, args.start)
    end_frame   = min(total, args.end if args.end else total)
    n_frames    = end_frame - start_frame + 1
    
    print(f"Processing frames {start_frame} – {end_frame}  ({n_frames} frames)")
    if use_correlated:
        print(f"Method: pure 3D->2D projection  |  cam_yaw/pitch=GLOBAL FRAME (from correlated CSV)  [OK]")
    else:
        print(f"Method: pure 3D->2D projection  |  cam_yaw=bearing(computed)  |  cam_pitch=gb_pitch")
    print()
    
    if args.debug:
        print("[DEBUG MODE] Printing detailed tracking info per frame")
        print("  Watch for: Xc (should change as drone moves left/right)")
        print()


    # ── Output ────────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (W, H))
    if not writer.isOpened(): sys.exit(f"[ERROR] Cannot create: {args.output}")

    # Initialize altitude noise filter
    alt_filter = AltitudeFilter(window_size=5)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
    t0=time.time()
    counts={'IN_FRAME':0,'OUT_OF_FRAME':0,'BEHIND_CAM':0,'NO_DATA':0}
    
    if args.debug:
        print("[DIAG] Starting frame processing loop...")
        sys.stdout.flush()

    frame_count = 0
    for frame_num in range(start_frame, end_frame+1):
        ret, img = cap.read()
        if not ret:
            print(f"\n[WARN] Video ended at frame {frame_num}")
            break

        frame_count += 1
        if args.debug and frame_count <= 3:
            print(f"[DIAG] Frame {frame_num}: Read from video - shape={img.shape}")
            sys.stdout.flush()

        srt_f = srt_data.get(frame_num)
        if srt_f is None:
            if args.debug and frame_count <= 3:
                print(f"[DIAG] Frame {frame_num}: No SRT data, writing raw frame")
                sys.stdout.flush()
            writer.write(img); counts['NO_DATA']+=1; continue

        # Get target position (use correlated CSV or position.csv)
        if use_correlated:
            # Convert V.SRT IST timestamp to UTC for lookup
            ts_ist = pd.Timestamp(srt_f['ts_ist'], tz='Asia/Kolkata')
            ts_utc = ts_ist.tz_convert('UTC')
            # Try EO (RGB) camera first, then fall back to IR
            evader = get_evader_position_from_correlated(df_corr, ts_utc, camera_type='EO')
            if evader is None:
                evader = get_evader_position_from_correlated(df_corr, ts_utc, camera_type='IR')
            if evader is None:
                evader = get_evader_position_from_correlated(df_corr, ts_utc, camera_type=None)
        else:
            evader = get_evader_position(df_pos, srt_f['ts_ist'], args.clock_offset)
        
        if evader is None:
            if args.debug and frame_count <= 3:
                print(f"[DIAG] Frame {frame_num}: No evader data, writing raw frame")
                sys.stdout.flush()
            writer.write(img); counts['NO_DATA']+=1; continue
        
        # Apply altitude noise filter
        evader['tgt_alt'] = alt_filter.filter(evader['tgt_alt'])
        
        if args.debug and frame_count <= 3:
            print(f"[DIAG] Frame {frame_num}: Got evader at ({evader['tgt_lat']:.6f}, {evader['tgt_lon']:.6f})")
            sys.stdout.flush()

        # Get FOV for this frame's focal length
        hfov, vfov = compute_fov(srt_f['focal'], sw, sh)

        out = annotate_frame(img, frame_num, srt_f, evader,
                             hfov, vfov, sw, sh, args.box_size,
                             cam_yaw_offset=args.cam_yaw_offset,
                             cam_pitch_offset=args.cam_pitch_offset,
                             yaw_distance_scale_24mm=args.yaw_distance_scale_24mm,
                             yaw_distance_scale_161mm=args.yaw_distance_scale_161mm,
                             pitch_offset_24mm=args.pitch_offset_24mm,
                             pitch_offset_106mm=args.pitch_offset_106mm,
                             pitch_offset_161mm=args.pitch_offset_161mm,
                             pitch_distance_scale_24mm=args.pitch_distance_scale_24mm,
                             pitch_distance_scale_161mm=args.pitch_distance_scale_161mm,
                             distance_ref=args.distance_ref,
                             auto_pitch=args.auto_pitch,
                             debug=args.debug,
                             use_global_angles=use_correlated,
                             smooth_alpha=args.smooth_alpha)
        
        if args.debug and frame_count <= 3:
            print(f"[DIAG] Frame {frame_num}: Annotate_frame completed")
            sys.stdout.flush()

        # Count status
        px,py,_ = project_3d_to_pixel(
            srt_f['obs_lat'],srt_f['obs_lon'],srt_f['obs_alt'],
            evader['tgt_lat'],evader['tgt_lon'],evader['tgt_alt'],
            haversine_bearing(srt_f['obs_lat'],srt_f['obs_lon'],
                              evader['tgt_lat'],evader['tgt_lon']),
            srt_f['gb_pitch'],
            srt_f['focal'],sw,sh,W,H)
        if px is None: counts['BEHIND_CAM']+=1
        elif 0<=px<W and 0<=py<H: counts['IN_FRAME']+=1
        else: counts['OUT_OF_FRAME']+=1

        writer.write(out)
        
        if args.debug and frame_count <= 3:
            print(f"[DIAG] Frame {frame_num}: Written to video")
            sys.stdout.flush()

        # Progress reporting - every 10 frames in debug mode, every 30 frames otherwise
        report_interval = 10 if args.debug else 30
        if frame_num % report_interval == 0 or frame_num == end_frame:
            el=time.time()-t0; done=frame_num-start_frame+1
            fp=done/el if el>0 else 1; eta=(n_frames-done)/fp
            pct=100*done/n_frames
            bar="#"*int(pct/2)+"-"*(50-int(pct/2))
            print(f"\r[{bar}]{pct:5.1f}% f{frame_num} {fp:.1f}fps ETA{eta:.0f}s "
                  f"IN={counts['IN_FRAME']} OUT={counts['OUT_OF_FRAME']} "
                  f"BEHIND={counts['BEHIND_CAM']}  ",
                  end="",flush=True)

    cap.release(); writer.release()
    el=time.time()-t0
    print(f"\n\nDone in {el:.1f}s")
    print(f"  IN_FRAME    : {counts['IN_FRAME']}")
    print(f"  OUT_OF_FRAME: {counts['OUT_OF_FRAME']} (box drawn orange at edge)")
    print(f"  BEHIND_CAM  : {counts['BEHIND_CAM']} (box drawn red at centre)")
    print(f"  NO_DATA     : {counts['NO_DATA']}")
    print(f"  Output: {args.output}")
    print()
    print("TRACKING DATA SOURCE:")
    if use_correlated:
        print("  [OK] Using CORRELATED CSV with GLOBAL FRAME camera angles")
        print("    > Horizontal tracking FIXED by using measured camera yaw/pitch!")
        print("    > Camera angles: primary_gb_yaw, primary_gb_pitch from correlated CSV")
    else:
        print("  [NO] Using legacy position.csv with computed bearing")
        print("    > For correct horizontal tracking, provide --correlated CSV")
    print()
    print("NOTE: OUT_OF_FRAME means the camera is NOT pointing at the evader.")
    print("  - Frames 1-3200: evader is directly overhead (vertical takeoff),")
    print("    camera points NNE (gb_yaw~26deg) -> evader is out of FOV")
    print("  - Frames 3991+: focal=161mm (HFOV=6.15deg, very narrow),")
    print("    evader is 4-7deg off camera centre -> outside narrow FOV")
    print("  - Frames 3200-3990 (24mm): evader IS in frame")
    print()
    print("LEFT/RIGHT TRACKING ISSUE?")
    print("  If box moves up/down but NOT left/right as drone moves:")
    print("  1. Run with --debug to see if Xc (camera X) is changing")
    print("  2. If Xc is NOT changing, gimbal might not have yaw data")
    print("  3. Check your V.SRT file for 'gimbal_yaw' field (not just 'gb_yaw')")
    print("  4. If it exists, update the regex in parse_vsrt() to extract gimbal_yaw")
    print("  5. Or use --cam-yaw-offset to manually calibrate gimbal pointing")
    print()
    print("CALIBRATION: If target is out-of-frame when it should be visible:")
    print("  1. Try --auto-pitch to use GPS elevation instead of gb_pitch")
    print("  2. Try --cam-yaw-offset ±10 to correct gimbal heading misalignment")
    print("  3. Try --cam-pitch-offset ±5 to correct pitch misalignment")
    print("  Example: python annotate_video.py ... --auto-pitch --cam-yaw-offset 5")
    print()
    print("CAMERA ANGLE OFFSET DETECTION (NEW - Global Frame Alignment):")
    print("  If box is always off to one side (e.g., always on right edge):")
    print("  1. Run with --debug to see: GPS bearing vs CSV yaw vs final_yaw")
    print("  2. If (GPS bearing - CSV yaw) is CONSISTENT offset, that's the angle error")
    print("  3. Calculate: offset_needed = GPS_bearing - CSV_yaw (from debug output)")
    print("  4. Example: If GPS=180deg, CSV=200deg -> offset = -20deg")
    print("  5. Test: python ... --cam-yaw-offset -20 --debug --start 1500 --end 1530")
    print("  6. Box should now move with drone left/right (HORIZONTAL TRACKING WORKS)")
    print()
    print("VERIFY CALIBRATION:")
    print("  - Watch Xc value in debug output - should CHANGE (positive/negative) as drone moves left/right")
    print("  - If Xc is STATIC, yaw offset is still wrong")
    print("  - If Xc CHANGES, yaw offset is CORRECT - box moves with drone!")
    print()
    print("To re-encode for Windows: "
          f"ffmpeg -i {args.output} -c:v libx264 -crf 18 -preset fast final.mp4")

if __name__=="__main__":
    main()