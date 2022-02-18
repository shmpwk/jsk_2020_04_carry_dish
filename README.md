# jsk_2020_04_carry_dish
Grasping dishes!

## Syetem overview
![icra2021-crop-2](https://user-images.githubusercontent.com/42209144/154614063-7344a572-d7fa-4bbf-87f1-32e25e5c881b.svg)

## Start now

```
mkdir my_ws/src -p
cd my_ws/src
git clone git@github.com:shmpwk/jsk_2020_04_carry_dish.git
catkin build
```

## Example demo in simulator
- Recognize, grasp, carry and put dish

### irt-viewer demo
```
roseus irt-demo.l
```

### kinematics simulator demo
```
roseus kinematics-demo.l
```

### gazebo demo
```
roslaunch common.launch
roseus pr2-tabletop-object-grasp-dual-success.l
```

## Grasp learning
### When training,

#### with real PR2 robot
- **[Here is sample collected data](https://drive.google.com/drive/folders/1TC3_E2abqi5bCzfpjCcFwmxzDIgBsDOj)**
```
$ roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch
$ roseus euslisp/grasp_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=false gui:=false
```

#### with simulator
```
$ roslaunch jsk_2020_4_carry_dish common.launch
$ roseus euslisp/grasp_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=true gui:=false
```

### Train the model
- **[Here is sample trained model](https://drive.google.com/drive/folders/1q6QLKSp9woM1LSmSZAF6cMOnG9Wt9mtz)**

When the input image is rgbd
```
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/rgbd_grasp_system.py
```
When the input image is rgb
```
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/rgb_grasp_system.py
```
or
```
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/rgb_grasp_system_2.py
```
which depends on the dataset size.

You can check training loss by
```
$ tensorboard --logdir="Data/loss/[YOUR_FOLDER]" --load_fast=false
```

![icra2021-crop-6](https://user-images.githubusercontent.com/42209144/154615025-68332814-d6a9-429f-a2ad-5667c13dd28d.svg)


### When inferrence,
```
$ roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch
$ roseus euslisp/test_grasp_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=false gui:=false
```
When using rgbd image,
```
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/test_rgbd_grasp_system.py 
```
When using rgb image,
```
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/test_rgb_grasp_system.py 
```

With moveit,
```
$ roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch
$ roseus euslisp/test_moveit_grasp_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=false gui:=false
$ python ~/my_ws/src/jsk_2020_04_carry_dish/scripts/test_multi_rgb_grasp_system_short.py
$ roslaunch jsk_pr2_startup start_pr2_moveit.launch USE_KINECT:=true USE_LASER_AND_KINECT:=false
$ roslaunch pr2_moveit_config moveit_rviz.launch config:=true
$ roslaunch wash_dish detect_dirt.launch
```
![icra2021-crop-7](https://user-images.githubusercontent.com/42209144/154615043-564296f7-0e7f-4487-89e0-bcbd7a14ad94.svg)


### Demo
- **[Here is sample rosbag](https://drive.google.com/drive/folders/1X-iUz-DcNSpKsJFQXUTS3t0vuNs6tjXQ)**
```
$ roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch
$ roseus euslisp/grasp_move_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=false gui:=false
```
After finishing tidying up, stop eus program and then execute
```
$ euslisp/pr2-tabletop-object-grasp-dual-tray-pick.l
```
