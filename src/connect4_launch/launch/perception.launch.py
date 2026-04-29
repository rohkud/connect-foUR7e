from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([

        # --- Calibration ---
        Node(
            package='board_calibration',
            executable='board_corners',
            name='board_corners',
            output='screen'
        ),

        Node(
            package='board_calibration',
            executable='disc_colors',
            name='disc_colors',
            output='screen'
        ),

        # --- Perception ---
        Node(
            package='board_detector',
            executable='board_node',
            name='board_detector',
            output='screen'
        ),

        Node(
            package='disc_detector',
            executable='disc_node',
            name='disc_detector',
            output='screen'
        ),

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
        # --- Transforms ---
        Node(
            package='planning',
            executable='static_tf',
            name='static_tf',
            output='screen'
        ),

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
        # --- Planning ---
        Node(
            package='game_planner',
            executable='game_planner_node',
            name='game_planner',
            output='screen'
        ),

    ])