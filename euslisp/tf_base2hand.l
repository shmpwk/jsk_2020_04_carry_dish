#!/usr/bin/env roseus

(ros::roseus "tf_listener")

(setq *l* (instance ros::transform-listener :init))
(setq target_frame "/r_gripper_tool_frame")
(setq local_frame "/base_footprint")
(setq target-local
      (send *l* :lookup-transform
            local_frame target_frame (ros::time 0)))
;;Already world coorginate?
;;(setq target-world
;;      (send target-local :transform
;;            (send *pr2* 

(ros::publish "/end_pos_rot" 
              instance geometry_msgs:: )

