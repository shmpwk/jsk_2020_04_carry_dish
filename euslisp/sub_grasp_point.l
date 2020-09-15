#!/usr/bin/env roseus
(defvar *grasp-target-bounding-box-topic* "/bounding_box_marker/selected_box")
(defvar *grasp-status-topic* "/tabletop_object_grasp_status")
(defvar *base-frame-id* "/base_footprint")

(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::roseus-add-msgs "geometry_msgs")
(ros::roseus "sub-grasp-point" :anonymous t)
;;(require "package://jsk_2020_4_carry_dish/euslisp/lib/calc.l")

;; sub-grasp-pointのcallback
(defun grasp-point-cb (msg)
  (print (list 'cb (send msg :pose :position :x)))
  (let ((cds (ros::tf-pose->coords (send msg :pose)))
        (base->parent (send *tfl* :lookup-transform
                            *base-frame-id* (send msg :header :frame_id)
                            (ros::time 0))))
    (send *tfb* :send-transform
          (send (send base->parent :copy-worldcoords)
                :transform cds)
          *base-frame-id* "target_object" (ros::time 0)
          )
    ;;(print (make-coords :pos (send base->parent :worldpos)))
    (print (make-coords :pos (send (send (send base->parent :copy-worldcoords)
                :transform cds) :worldpos)
                        :rpy (float-vector 0 0 (deg2rad 90))))
  )
)

(defun sub-grasp-point ()
   (ros::subscribe "grasp_point" geometry_msgs::posestamped #'grasp-point-cb)
   )

;; 把持点をsubscribe
(setq *tfl* (instance ros::transform-listener :init))
(setq *tfb* (instance ros::transform-broadcaster :init))
(ros::advertise "/grasp_state" std_msgs::String 1)
(sub-grasp-point)

(ros::spin)

