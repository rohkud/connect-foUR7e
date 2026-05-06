# connect-foUR7e
https://docs.google.com/document/d/1DzwP1deeW31QpVgI9eP0ryOaBIvRIhxhT5L1bDk_k2U/edit?tab=t.0

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
source install/setup.bash
ros2 launch usb_cam camera.launch.py
```

Pick corners
```
source install/setup.bash
ros2 run board_calibration board_corners
```

Pick disc colors
```
source install/setup.bash
ros2 run board_calibration disc_colors
```

Launch the game logic
```
source install/setup.bash
ros2 launch connect4_launch perception.launch.py
```

Launch moveit and ik service
```
source install/setup.bash
ros2 launch planning lab5_bringup.launch.py
```

## Testing

### Tuck
```
ros2 run ur7e_utils reset_state && ros2 run ur7e_utils tuck
```

### To setup connect4 board

```
source install/setup.bash
ros2 run planning debug
```
