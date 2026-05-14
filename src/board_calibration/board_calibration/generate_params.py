import numpy as np

# -------------------------
# Load calibration
# -------------------------
K = np.load("K.npy").reshape(3, 3)
dist = np.load("dist.npy").flatten()

fx, fy = float(K[0, 0]), float(K[1, 1])
cx, cy = float(K[0, 2]), float(K[1, 2])

image_width = 1280
image_height = 720

# -------------------------
# Projection matrix
# -------------------------
P = [
    fx, 0.0, cx, 0.0,
    0.0, fy, cy, 0.0,
    0.0, 0.0, 1.0, 0.0
]

# -------------------------
# Write raw ROS YAML text
# -------------------------
output_file = "camera1.yaml"

with open(output_file, "w") as f:
    f.write(f"image_width: {image_width}\n")
    f.write(f"image_height: {image_height}\n")
    f.write("camera_name: camera1\n\n")

    f.write("camera_matrix:\n")
    f.write("  rows: 3\n")
    f.write("  cols: 3\n")
    f.write("  data: [")
    f.write(", ".join(map(str, K.flatten().tolist())))
    f.write("]\n\n")

    f.write("distortion_model: plumb_bob\n\n")

    f.write("distortion_coefficients:\n")
    f.write("  rows: 1\n")
    f.write(f"  cols: {len(dist)}\n")
    f.write("  data: [")
    f.write(", ".join(map(str, dist.tolist())))
    f.write("]\n\n")

    f.write("rectification_matrix:\n")
    f.write("  rows: 3\n")
    f.write("  cols: 3\n")
    f.write("  data: [1, 0, 0, 0, 1, 0, 0, 0, 1]\n\n")

    f.write("projection_matrix:\n")
    f.write("  rows: 3\n")
    f.write("  cols: 4\n")
    f.write("  data: [")
    f.write(", ".join(map(str, P)))
    f.write("]\n")

print(f"Wrote {output_file}")