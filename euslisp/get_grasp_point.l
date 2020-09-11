#!/usr/bin/env roseus

(ros::roseus-add-srvs "roseus")
(ros::roseus "get-grasp-point")
(ros::load-ros-manifest "jsk_recognition_msgs")

(defun get-grasp-point ()
  (ros::wait-for-service "get-grasp-point")
  (setq req (instance jsk_recognition_msgs::BoundingBoxArray :init))
  (send req :a (random 10))
  (setq before (ros::time-now))
  (setq res (ros::service-call "get-grasp-point" req))
  (setq after (ros::time-now))
  (format t "~d" (send res))
  )

(get-grasp-point)
