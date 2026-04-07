# connect-foUR7e


```
ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true
\rgb_camera.color_profile:=1920x1080x30
```

## Startup

Turn on robot
```
ros2 run ur7e_utils enable_comms
```

Launch aruco tag detection
```
source install/setup.bash
ros2 launch visual_servoing lab7.launch.py
```

Launch block detection
```
source install/setup.bash
ros2 launch planning lab5_bringup.launch.py
```

Separate terminal
```
source install/setup.bash
ros2 run planning main
```

## Testing

### Tuck
```
ros2 run ur7e_utils reset_state && ros2 run ur7e_utils tuck
``
