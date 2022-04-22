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
            launch.actions.DeclareLaunchArgument(
                name="rgb_camera_topic", default_value="/rgb_image"
            ),
            launch.actions.DeclareLaunchArgument(
                name="ground_mask", default_value="[0, 0, 80, 255, 50, 200]"
            ),
            launch.actions.DeclareLaunchArgument(
                name="grass_mask", default_value="[43,50,20,128,255,250]"
            ),
            launch.actions.DeclareLaunchArgument(
                name="horizon_pct", default_value="0.25"
            ),
            launch.actions.DeclareLaunchArgument(
                name="morph_kernel_size", default_value="30"
            ),
            launch_ros.actions.Node(
                package="lane_detection_indy",
                executable="road_mask_node",
                name="road_mask_node",
                parameters=[
                    {
                        "rgb_camera_topic": launch.substitutions.LaunchConfiguration(
                            "rgb_camera_topic"
                        ),
                        "debug": launch.substitutions.LaunchConfiguration("debug"),
                        "morph_kernel_size": launch.substitutions.LaunchConfiguration(
                            "morph_kernel_size"
                        ),
                        "ground_mask": launch.substitutions.LaunchConfiguration(
                            "ground_mask"
                        ),
                        "grass_mask": launch.substitutions.LaunchConfiguration(
                            "grass_mask"
                        ),
                        "horizon_pct": launch.substitutions.LaunchConfiguration(
                            "horizon_pct"
                        ),
                    }
                ],
            ),
        ]
    )
