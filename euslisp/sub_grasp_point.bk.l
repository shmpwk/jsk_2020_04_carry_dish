#!/usr/bin/env roseus

(ros::roseus-add-msgs "geometry_msgs")
(ros::roseus "sub-grasp-point" :anonymous t)

(defun grasp-point-cb (msg)
  (print (list 'cb (send msg :position :x)))
  )

(ros::subscribe "grasp_point" geometry_msgs::pose #'grasp-point-cb)
(ros::spin)

