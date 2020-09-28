#!/usr/bin/env roseus

(ros::roseus-add-srvs "roseus")
(ros::roseus "get-grasp-point")
(ros::load-ros-manifest "jsk_recognition_msgs")

(defun get-grasp-point ()
  (ros::wait-for-service "/data_collection/save_request")
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq before (ros::time-now))
  (setq res (ros::service-call "/data_collection/save_request" req))
  (setq after (ros::time-now))
  (format t "~d" (send res))
  )

(get-grasp-point)
