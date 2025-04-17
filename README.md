![12-10-1](https://github.com/user-attachments/assets/c0d93939-f46f-4426-af62-79f680ef4ca3)


## Manuls
1. The Code is running on the Jetson Orin on GO2 and is controlled with the unitree joystick.
2. Install ROS2 foxy and the [unitree ros package for Go2](https://support.unitree.com/home/en/developer/ROS2_service).
3. Create a Python virtual env and install the dependencies.
    - Install pytorch on a Python 3 environment.
        ```bash
        conda create -n freq-hardware python==3.8
        conda activate freq-hardware
        ```
    
    - Download the pip wheel file from [here](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) with v1.11.0 of JetPack 5. Then install it with
        ```bash
        pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl
        ```

    - Install `ros2-numpy` from [here](https://github.com/nitesh-subedi/ros2_numpy) in a new colcon_ws, where you prefer.
        ```bash
        pip install transformations pybase64
        mkdir -p ros2_numpy_ws/src
        cd ros2_numpy_ws/src
        git clone https://github.com/nitesh-subedi/ros2_numpy.git
        cd ../
        colcon build
        ```
        
4. Run the Controller.
    - Put the robot on the ground, power on the robot, and **turn off the builtin sport service**.
    - Launch a terminal onboard. Activate the Python environment, source the Unitree Ros workspace and ros2_numpy_ws workspace. 
    - ```python go2_run_freq_interace_ab.py```
    - Currently, the robot will not actually move its motors. You may see the ros topics. If you want to let the robot move, you can add the argument `--nodryrun` in the command line, but **be careful**.
    - User can switch the finite state machine with the buttons and control the commanded velocity with the joystick.




