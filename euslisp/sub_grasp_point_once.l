#!/usr/bin/env roseus
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::roseus-add-msgs "geometry_msgs")
(require "models/arrow-object.l")

(require :pr2-interface "package://pr2eus/pr2-interface.l")

(defun sub-grasp-point ()
  (let* ((msg (one-shot-subscribe "grasp_point" geometry_msgs::PoseStamped
                                  :after-stamp (ros::time 0)))
         (cds (when msg (ros::tf-pose->coords (send msg :pose))))
         (base->parent
           (when msg (send *tfl* :lookup-transform
                           *base-frame-id* (send msg :header :frame_id)
                           (ros::time 0)))))
    (print "aaaaaaaaaaaaaaaaaaaaaaaa")
    (when (and cds base->parent)
      (send (send base->parent :copy-worldcoords) :transform cds))))


(defun bounding-box->cube (msg)
  (let ((cds (ros::tf-pose->coords (send msg :pose)))
        (d (vector3-> 'list (send msg :dimensions)))
        (base->parent (send *tfl* :lookup-transform
                            *base-frame-id* (send msg :header :frame_id)
                            (ros::time 0))))
    (send *tfb* :send-transform
          (send (send base->parent :copy-worldcoords)
                :transform cds)
          *base-frame-id* "target_object" (send msg :header :stamp))
    (send (apply #'make-cube d)
          :transform
          (send (send base->parent :copy-worldcoords)
                :transform cds))
                )
  )


      (send *pr2* *rarm* :inverse-kinematics
            ;;(make-coords :pos target-point :worldpos)
            (make-coords :pos (v+ (send target-point :worldpos)
                                  (float-vector -100 0  0))
                         :rpy (v- (send target-point :worldpos)
                                  (send bounding-box->cube msg))
            :rotation-axis :t))


