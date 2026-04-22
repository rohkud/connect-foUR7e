from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='game_planner',
            executable='game_planner',
            name='game_planner',
            output='screen',
            parameters=[
                {'player_color': 'red'},
                {'solve_interval': 3.0},
                {'alpha': 0.3},
            ],
        ),
    ])
