from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessStart
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    
    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="true")

    return LaunchDescription([

        # --- Perception ---
        Node(
            package='board_detector',
            executable='board_node',
            name='board_detector',
            output='screen'
        ),

        Node(
            package='board_localizer',
            executable='board_localizer',
            name='board_localizer',
            output='screen'
        ),

        # --- Pixel → 3D Service ---
        Node(
            package='piece_localization',
            executable='piece_localizer',
            name='piece_localization',
            output='screen'
        ),

        # --- Game Logic ---
        Node(
            package='game_state',
            executable='game_state_node',
            name='game_state',
            output='screen'
        ),

        Node(
            package='game_solver',
            executable='game_solver_node',
            name='game_solver',
            output='screen'
        ),

        Node(
            package='game_planner',
            executable='game_planner',
            name='game_planner',
            output='screen'
        ),

        # --- Transforms ---
        # Node(
        #     package='planning',
        #     executable='static_tf',
        #     name='static_tf',
        #     output='screen'
        # ),

        Node(
            package='planning',
            executable='camera_tf',
            name='camera_tf',
            output='screen'
        ),

        # --- IK Solver ---
        Node(
            package='planning',
            executable='ik',
            name='ik',
            output='screen'
        ),

        # --- Control / Execution ---
        Node(
            package='game_planner',
            executable='main',
            name='game_planner_main',
            output='screen'
        ),

        # --- MoveIt ---
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory("ur_moveit_config"),
                    "launch",
                    "ur_moveit.launch.py"
                )
            ),
            launch_arguments={
                "ur_type": ur_type,
                "launch_rviz": launch_rviz
            }.items(),
        ),
    ])