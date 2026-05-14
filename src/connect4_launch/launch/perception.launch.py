import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")
    aruco_params = os.path.join(
        get_package_share_directory("connect4_launch"),
        "config",
        "perception_parameters.yaml",
    )
    camera_tf_params = os.path.join(
        get_package_share_directory("connect4_launch"),
        "config",
        "perception_parameters.yaml",
    )

    return LaunchDescription(
        [
            # --- Perception ---
            Node(
                package="disc_detector",
                executable="disc_node",
                name="disc_detector",
                output="screen",
            ),
            # --- Pixel → 3D Service ---
            Node(
                package="game_state",
                executable="localizer",
                name="localization_service",
                output="screen",
            ),
            # --- Game Logic ---
            Node(
                package="game_state",
                executable="disc_state",
                name="disc_state",
                output="screen",
            ),
            Node(
                package="game_state",
                executable="board_state",
                name="board_state",
                output="screen",
            ),
            Node(
                package="game_state",
                executable="solver",
                name="solver_node",
                output="screen",
            ),
            Node(
                package="game_planner",
                executable="game_planner",
                name="game_planner",
                output="screen",
            ),
            Node(
                package="connect4_launch",
                executable="aruco_node",
                name="aruco_node",
                parameters=[aruco_params],
            ),
            Node(
                package="connect4_launch",
                executable="camera_tf",
                name="camera_tf",
                output="screen",
                parameters=[camera_tf_params],
            ),
            # --- Control / Execution ---
            Node(
                package="game_planner",
                executable="main",
                name="game_planner_main",
                output="screen",
            ),
            Node(
                package="planning",
                executable="main",
                name="planning_main",
                output="screen",
            ),
        ]
    )

