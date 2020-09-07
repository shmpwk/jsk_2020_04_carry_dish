#!/usr/bin/env roseus
;; -*- coding: utf-8 -*-

(load "package://pr2eus/pr2-interface.l")

(setq *ri* (instance pr2-interface :init))
(setq *robot* (pr2))
(send *robot* :init)

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")

(ros::roseus "pr2_tabletop_object_grasp_node")
(ros::rate 10)

(require :pr2-interface "package://pr2eus/pr2-interface.l")

(defun pr2-pregrasp-pose ()
  (send *pr2* :reset-manip-pose)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation))


(defun data-collection ()
  (setq is-bimanual false)

  ;;data collection configs
  (setq localtime (unix::localtime))
  (setq date (format nil "~A~0,2d~0,2d"
                     (+ (elt localtime 5) 1900) (+ (elt localtime 4) 1) (elt localtime 3)))
  (setq save-dir
      (format nil "~A_~A_grasp/trial_~A_~0,2d~0,2d~0,2d"
              date (if is-bimanual "dualarm" "singlearm")
              date (elt localtime 2) (elt localtime 1) (elt localtime 0)))

   (ros::set-dynparam
     (format nil "/before_grasp_data_collection")
     (cons "save_dir" (format nil "~A/~A" save-dir "before_grasp")))


   (if (send
         (ros::service-call (format nil "/data_collection/start_request")
                             (instance std_srvs::TriggerRequest :init))
          :success)
      (ros::ros-info "[collect-data-before-grasp] success to start collecting data")
      (ros::ros-error "[collect-data-before-grasp] failed to start collecting data"))
   )

