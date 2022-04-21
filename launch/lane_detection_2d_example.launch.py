from launch import LaunchDescription
import os
from ament_index_python.packages import get_package_share_directory
import launch_ros
from pathlib import Path
import launch


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory("lane_detection_indy"))
    video_path = Path(base_path) / "configs" / "P6010001.MOV"
    assert video_path.exists(), f"[{video_path}] does not exist"
    rviz_path = base_path + "/configs/example.rviz"
    return LaunchDescription(
        [
            launch_ros.actions.Node(
                package="lane_detection_indy",
                executable="road_mask_node",
                name="road_mask_node",
                parameters=[{"rgb_camera_topic": "/rgb_image", "debug": True}],
            ),
            launch_ros.actions.Node(
                package="lane_detection_indy",
                executable="video_player_node",
                name="video_player_node",
                parameters=[{"video_path": video_path.as_posix(), "debug": False}],
            ),
            launch_ros.actions.Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_path],
            ),
        ]
    )
