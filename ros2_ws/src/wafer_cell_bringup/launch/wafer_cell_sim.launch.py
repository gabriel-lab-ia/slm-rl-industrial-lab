from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='wafer_cell_bringup',
            executable='wafer_cell_sim',
            name='wafer_cell_sim',
            output='screen',
        )
    ])
