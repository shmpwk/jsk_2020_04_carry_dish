#!/usr/bin/env roseus

(ros::roseus-add-msgs "geometry_msgs")
(ros::roseus "sub-grasp-point" :anonymous t)
;;(require "package://jsk_2020_4_carry_dish/euslisp/lib/calc.l")

(defun grasp-point-cb (msg)
  ;;(print (list 'cb (send msg :position :x)))
  (setq a (ros::tf-pose->coords msg))
  ()
  (print a)
  )

(ros::subscribe "grasp_point" geometry_msgs::pose #'grasp-point-cb)
(ros::spin)

