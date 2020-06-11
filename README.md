# udrone_boat_localization

This package requires Python 3 and a separate ROS catkin workspace. 
Follow these steps to create the catkin workspace with Python3.5 (ROS Kinetic and Ubuntu 16.04)

### Setup Python3 enabled catkin workspace


``` bash
mkdir $HOME/rospy3_kinetic
cd $HOME/rospy3_kinetic
sudo apt install python3-venv libpython3-dev python-catkin-tools
python3.5 -m venv venv3.5_ros
source venv3.5_ros/bin/activate
pip3 install wheel empy defusedxml netifaces
pip3 install -U rosdep rosinstall_generator wstool rosinstall
rosinstall_generator --deps --tar --rosdistro=kinetic ros_base > ros_base.rosinstall
wstool init src -j8 ros_base.rosinstall
sudo rosdep init
rosdep update
rosdep check --from-paths src/ -i
catkin build -DCATKIN_ENABLE_TESTING=0 -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3.5
source devel/setup.bash
```

NOTE: It is important that python3 virtualenv is activated and setup.bash is sourced in any terminals that uses the scripts in the repository.

### Install filterpy package

``` bash
cd $HOME/rospy3_kinetic
source venv3.5_ros/bin/activate
source devel/setup.bash
pip install matplotlib==2.2.0 numpy filterpy

```

### Add geometry and geometry2 packages to rospy3_kinetic workspace

``` bash
cd $HOME/rospy3_kinetic
source venv3.5_ros/bin/activate
source devel/setup.bash
cd $HOME/rospy3_kinetic/src
git clone https://github.com/ros/geometry.git
git clone https://github.com/ros/geometry2.git
cd geometry
git checkout indigo-devel
cd ../geometry2
git checkout indigo-devel
```

