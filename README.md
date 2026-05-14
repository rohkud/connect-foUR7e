# connect-foUR7e
CV Connect 4 using UR7e arm. 

#### Requires
- UR7e
- Logitech C922 Pro Stream Webcam

#### Features
- Board Localization (using homography correction)
- Disc color detection
- Connect-4 Game Solver
- IK and motion planning for UR7e
- Autonomous disc pickup and placement

## Repository Structure

```text
src/
├── board_calibration/
├── connect4_launch/
├── disc_detector/
├── game_planner/
├── game_state/
├── planning/
├── piece_localization_interfaces/
├── ros2_aruco/
└── usb_cam/
```

## Prerequisites
### Set up distrobox (1st time EVER)
```
inst-containers-setup
~ee106a/create-ros2-container
```

## Startup
### Enter distrobox
```
distrobox enter ros2
```

## Robot Bringup
### Enable UR7e communication
```
ros2 run ur7e_utils enable_comms
```

### Launch USB camera
```
source install/setup.bash
ros2 launch usb_cam camera.launch.py
```

## Calibration
### Select board corners
```
source install/setup.bash
ros2 run board_calibration board_corners
```

### Select disc colors
```
source install/setup.bash
ros2 run board_calibration disc_colors
```
## Launch perception and game logic
```
source install/setup.bash
ros2 launch connect4_launch perception.launch.py
```

## Motion planning
### Launch Moveit and IK service
```
source install/setup.bash
ros2 launch planning lab5_bringup.launch.py
```

## Utilities

### Tuck Robot
```
ros2 run ur7e_utils reset_state && ros2 run ur7e_utils tuck
```

### Debug IK or CV using known board + piece locations

```
ros2 run planning debug
```
