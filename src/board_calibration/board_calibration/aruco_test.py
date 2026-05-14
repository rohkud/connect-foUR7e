import cv2
import cv2.aruco as aruco
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Image
from std_msgs.msg import Header


class ArucoCompareNode(Node):

    def __init__(self):
        super().__init__("aruco_compare_node")

        # =========================================================
        # LOAD CALIBRATION
        # =========================================================
        self.K_new = np.load("src/board_calibration/board_calibration/K.npy")

        self.dist_new = np.load("src/board_calibration/board_calibration/dist.npy")

        self.K_old = np.array(
            [
                [997.1410709758351, 0, 620.2712277820584],
                [0, 1000.62021150938, 401.6358113787741],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        self.dist_old = np.array(
            [0.075217, -0.136408, 0.0111199, -0.0016369, 0.0],
            dtype=np.float64,
        )

        # =========================================================
        # ARUCO
        # =========================================================
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

        self.params = aruco.DetectorParameters_create()

        if hasattr(aruco, "ArucoDetector"):
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.params)
            self.use_new = True
        else:
            self.use_new = False

        # =========================================================
        # MARKER MODEL
        # =========================================================
        marker_length = 0.16
        half = marker_length / 2

        self.obj_points = np.array(
            [
                [-half, half, 0],
                [half, half, 0],
                [half, -half, 0],
                [-half, -half, 0],
            ],
            dtype=np.float64,
        )

        # =========================================================
        # ROS
        # =========================================================
        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            "/camera1/image_raw",
            self.image_callback,
            10,
        )

        # NEW intrinsics pose array
        self.pose_pub_new = self.create_publisher(
            PoseArray,
            "/aruco_markers_new",
            10,
        )

        # OLD intrinsics pose array
        self.pose_pub_old = self.create_publisher(
            PoseArray,
            "/aruco_markers_old",
            10,
        )

        self.get_logger().info("Aruco compare node running")

    # =============================================================
    # DETECT MARKERS
    # =============================================================
    def detect(self, gray):

        if self.use_new:
            return self.detector.detectMarkers(gray)

        return aruco.detectMarkers(
            gray,
            self.aruco_dict,
            parameters=self.params,
        )

    # =============================================================
    # RVEC -> QUATERNION
    # =============================================================
    def rvec_to_quaternion(self, rvec):

        rot_matrix, _ = cv2.Rodrigues(rvec)

        quat = R.from_matrix(rot_matrix).as_quat()

        # scipy order:
        # x y z w
        return quat

    # =============================================================
    # IMAGE CALLBACK
    # =============================================================
    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.detect(gray)

        # =========================================================
        # POSE ARRAY (NEW)
        # =========================================================
        pose_array_new = PoseArray()

        pose_array_new.header = Header()
        pose_array_new.header.stamp = self.get_clock().now().to_msg()

        pose_array_new.header.frame_id = "camera_link"

        # =========================================================
        # POSE ARRAY (OLD)
        # =========================================================
        pose_array_old = PoseArray()

        pose_array_old.header = Header()
        pose_array_old.header.stamp = self.get_clock().now().to_msg()

        pose_array_old.header.frame_id = "camera_link"

        # =========================================================
        # NO MARKERS
        # =========================================================
        if ids is None or len(ids) == 0:

            self.pose_pub_new.publish(pose_array_new)

            self.pose_pub_old.publish(pose_array_old)

            cv2.imshow("Aruco Compare", frame)

            cv2.waitKey(1)

            return

        # =========================================================
        # PROCESS MARKERS
        # =========================================================
        for c in corners:

            img_points = c.reshape(4, 2).astype(np.float64)

            # -----------------------------------------------------
            # NEW CALIBRATION
            # -----------------------------------------------------
            ok_new, rvec_new, tvec_new = cv2.solvePnP(
                self.obj_points,
                img_points,
                self.K_new,
                self.dist_new,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            # -----------------------------------------------------
            # OLD CALIBRATION
            # -----------------------------------------------------
            ok_old, rvec_old, tvec_old = cv2.solvePnP(
                self.obj_points,
                img_points,
                self.K_old,
                self.dist_old,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )

            # =====================================================
            # NEW POSE
            # =====================================================
            if ok_new:

                pose_new = Pose()

                pose_new.position.x = float(tvec_new[0])

                pose_new.position.y = float(tvec_new[1])

                pose_new.position.z = float(tvec_new[2])

                qx, qy, qz, qw = self.rvec_to_quaternion(rvec_new)

                pose_new.orientation.x = float(qx)
                pose_new.orientation.y = float(qy)
                pose_new.orientation.z = float(qz)
                pose_new.orientation.w = float(qw)

                pose_array_new.poses.append(pose_new)

            # =====================================================
            # OLD POSE
            # =====================================================
            if ok_old:

                pose_old = Pose()

                pose_old.position.x = float(tvec_old[0])

                pose_old.position.y = float(tvec_old[1])

                pose_old.position.z = float(tvec_old[2])

                qx, qy, qz, qw = self.rvec_to_quaternion(rvec_old)

                pose_old.orientation.x = float(qx)
                pose_old.orientation.y = float(qy)
                pose_old.orientation.z = float(qz)
                pose_old.orientation.w = float(qw)

                pose_array_old.poses.append(pose_old)

            # =====================================================
            # VISUALIZATION
            # =====================================================
            if ok_new and ok_old:

                z_new = float(tvec_new[2])
                z_old = float(tvec_old[2])

                # -------------------------------------------------
                # DRAW AXES
                # -------------------------------------------------
                cv2.drawFrameAxes(
                    frame,
                    self.K_new,
                    self.dist_new,
                    rvec_new,
                    tvec_new,
                    0.05,
                )

                overlay = frame.copy()

                cv2.drawFrameAxes(
                    overlay,
                    self.K_old,
                    self.dist_old,
                    rvec_old,
                    tvec_old,
                    0.05,
                )

                frame = cv2.addWeighted(
                    frame,
                    1.0,
                    overlay,
                    0.5,
                    0,
                )

                # -------------------------------------------------
                # INTRINSIC CENTERS
                # -------------------------------------------------
                cx_new = int(self.K_new[0, 2])
                cy_new = int(self.K_new[1, 2])

                cx_old = int(self.K_old[0, 2])
                cy_old = int(self.K_old[1, 2])

                # draw principal points
                cv2.circle(
                    frame,
                    (cx_new, cy_new),
                    8,
                    (0, 255, 0),
                    -1,
                )

                cv2.circle(
                    frame,
                    (cx_old, cy_old),
                    8,
                    (0, 140, 255),
                    -1,
                )

                # -------------------------------------------------
                # PROJECT MARKER ORIGIN
                # -------------------------------------------------
                origin_3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

                # NEW projection
                proj_new, _ = cv2.projectPoints(
                    origin_3d,
                    rvec_new,
                    tvec_new,
                    self.K_new,
                    self.dist_new,
                )

                px_new = int(proj_new[0][0][0])
                py_new = int(proj_new[0][0][1])

                # OLD projection
                proj_old, _ = cv2.projectPoints(
                    origin_3d,
                    rvec_old,
                    tvec_old,
                    self.K_old,
                    self.dist_old,
                )

                px_old = int(proj_old[0][0][0])
                py_old = int(proj_old[0][0][1])

                # -------------------------------------------------
                # DRAW VECTORS
                # -------------------------------------------------

                # NEW vector
                cv2.line(
                    frame,
                    (cx_new, cy_new),
                    (px_new, py_new),
                    (0, 255, 0),
                    3,
                )

                # OLD vector
                cv2.line(
                    frame,
                    (cx_old, cy_old),
                    (px_old, py_old),
                    (0, 140, 255),
                    3,
                )

                # endpoint circles
                cv2.circle(
                    frame,
                    (px_new, py_new),
                    6,
                    (0, 255, 0),
                    -1,
                )

                cv2.circle(
                    frame,
                    (px_old, py_old),
                    6,
                    (0, 140, 255),
                    -1,
                )

                # -------------------------------------------------
                # LABELS
                # -------------------------------------------------
                cv2.putText(
                    frame,
                    f"NEW Z: {z_new:.3f} m",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"OLD Z: {z_old:.3f} m",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 140, 255),
                    2,
                )

                # labels for principal points
                cv2.putText(
                    frame,
                    "NEW PP",
                    (cx_new + 10, cy_new),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    "OLD PP",
                    (cx_old + 10, cy_old),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 140, 255),
                    2,
                )

                print(f"NEW: {z_new:.3f} | " f"OLD: {z_old:.3f}")

        # =========================================================
        # PUBLISH
        # =========================================================
        self.pose_pub_new.publish(pose_array_new)

        self.pose_pub_old.publish(pose_array_old)

        cv2.imshow("Aruco Compare", frame)

        cv2.waitKey(1)


# ================================================================
# MAIN
# ================================================================
def main():

    rclpy.init()

    node = ArucoCompareNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    node.destroy_node()

    cv2.destroyAllWindows()

    rclpy.shutdown()


if __name__ == "__main__":
    main()

