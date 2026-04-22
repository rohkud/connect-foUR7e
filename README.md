# connect-foUR7e


```
<!-- ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
\rgb_camera.color_profile:=1920x1080x30 -->
```

## Startup

Get into distrobox (1st time EVER)
```
inst-containers-setup
~ee106a/create-ros2-container
distrobox enter ros2
```

Turn on robot
```
ros2 run ur7e_utils enable_comms
```

Launch usb cam
```
ros2 launch usb_cam camera.launch.py
```

Launch disc detection
```
ros2 run disc_detector disc_node
```

Pick corners
```
ros2 run board_detector board_corners
```

Run state detector
```
ros2 run game_state game_state_node
```

Publish the corners
```
ros2 run board_detector board_node
```

Separate terminal
```
source install/setup.bash
ros2 run planning main
```


Launch aruco tag detection
```
source install/setup.bash
ros2 launch visual_servoing lab7.launch.py
```

## Testing

### Tuck
```
ros2 run ur7e_utils reset_state && ros2 run ur7e_utils tuck
```
