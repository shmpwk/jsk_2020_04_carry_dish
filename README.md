# jsk_2020_04_carry_dish
Grasping dishes!

## Syetem overview
![icra2021-crop-2](https://user-images.githubusercontent.com/42209144/154614063-7344a572-d7fa-4bbf-87f1-32e25e5c881b.svg)

## Start now

```
$ sudo apt-get install ros-melodic-jskeus
$ source /opt/ros/melodic/setup.bash
```


## Example
- 皿を認識
- 皿を掴む
- 棚に皿を運ぶ
- 棚に置く

- irt-viewerのdemo
```
roseus irt-demo.l
```

- kinematics simulatorのdemo
```
roseus kinematics-demo.l
```

- gazeboで皿を掴むdemo
```
roslaunch common.launch
roseus pr2-tabletop-object-grasp-dual-success.l
```

  - common.launch 
    - pr2_tabletop_scene.launch
      - empty_world.launch
      - tabletop.world
    - pr2_tabletop.launch
      - tabletop_object_detector.launch

## grasp learning
### When training,
#### with real PR2 robot
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
```
$ roslaunch jsk_2020_4_carry_dish realpr2_tabletop.launch
$ roseus euslisp/grasp_move_sequence.l 
$ roslaunch jsk_2020_4_carry_dish gazebo_clicked_box_edge.launch use_sim:=false gui:=false
```
After finishing tidying up, stop eus program and then execute
```
$ euslisp/pr2-tabletop-object-grasp-dual-tray-pick.l
```
