import glob

import cv2
import cv2.aruco as aruco
import numpy as np

# --- config ---
image_pattern = "images/*.jpg"

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# 4.6 SAFE API (IMPORTANT)
board = aruco.CharucoBoard_create(
    5, 7, 0.037, 0.028, aruco_dict
)

# detector params (4.6 stable)
params = aruco.DetectorParameters_create()

all_corners = []
all_ids = []
image_size = None

images = glob.glob(image_pattern)

if len(images) == 0:
    raise RuntimeError("No images found!")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4.6-safe detection (avoid relying on ArUcoDetector class)
    corners, ids, _ = aruco.detectMarkers(
        gray, aruco_dict, parameters=params
    )

    if ids is None or len(ids) == 0:
        continue

    ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
        corners, ids, gray, board
    )

    # IMPORTANT: guard for 4.6 instability
    if charuco_ids is None or charuco_corners is None:
        continue

    if ret is None or ret < 10:
        continue

    all_corners.append(charuco_corners)
    all_ids.append(charuco_ids)

    image_size = (gray.shape[1], gray.shape[0])

if len(all_corners) == 0:
    raise RuntimeError("No valid Charuco detections found.")

# --- calibration (4.6 compatible call) ---
ret, K, dist, rvecs, tvecs = aruco.calibrateCameraCharuco(
    charucoCorners=all_corners,
    charucoIds=all_ids,
    board=board,
    imageSize=image_size,
    cameraMatrix=None,
    distCoeffs=None,
)

print("\n=== RESULTS ===")
print("RMS:", ret)
print("K:\n", K)
print("dist:\n", dist)

np.save("K.npy", K)
np.save("dist.npy", dist)