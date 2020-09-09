#!/usr/bin/env roseus
;; Author: Shumpei Wakabayashi <wakabayashi@jsk.imi.i.u-tokyo.ac.jp>
;; for all learning grasp system

(defun pr2-pregrasp-pose ()
  (send *pr2* :reset-manip-pose)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation))

;; service callでdepthをcsvに落とす
(defun get-depth ()
  )

;;bounding boxの表示？もともとできているのか


;; service callで把持候補のpclをcsvに落とす
(defun ()
  )

;; 把持点をsubし、graspする
(defun sub-grasp-point()
  )

;; 対象物を持ち上げられたか否か判断する
(defun judge-grasp-success ()
  ;;service call?
  )

(defun grasp-target-callback (msg)
  (let ((target-point (bounding-box->cube msg)))
    
    (get-depth)
    (sub-grasp-point)
    
    
    (if *grasping-object-p*
        (progn
          (publish-warn "cancel grasping...")
          (send *ri* :stop-motion)
          (send *ri* :stop-grasp *rarm* :wait t)
          (setq *grasping-object-p* nil))
      (setq *grasping-object-p* t))
    (unless
	(progn
	  (send *pr2* *rarm* :inverse-kinematics
		(make-coords :pos (v+ (send target-point :worldpos)
				       (float-vector 0 0 0))
			     :rpy #f(1.54 1.37 -1.54)) ;;どれか一つをランダムに
		:rotation-axis :t)
      (publish-error "out of reach...")
      (return-from grasp-target-callback nil))
    (progn
	  (publish-info "openning gripper...")
	  (print (send target-obj :worldpos))
	  (print  *d_xyz*)
	  )
    (setq *grasp-state* "reaching")
    (publish-grasp-state)
    (publish-info "pre grasp pose...")
    (send *ri* :stop-grasp *rarm* :wait t)
    (send *ri* :wait-interpolation)
    ;;(send *pr2* :head :look-at-hand *rarm*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 6000)
    (send *ri* :wait-interpolation)

    (setq *grasp-state* "grasp-approaching")
    (publish-grasp-state)
    (publish-info "reaching...")
    (send *pr2* *rarm* :move-end-pos #f(105 0 -5) :world)
    ;;(send *pr2* :head :look-at-hand *rarm*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (send *pr2* *rarm* :move-end-pos #f(0 0 -3) :world)
    ;;(send *pr2* :head :look-at-hand *rarm*)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    (send *ri* :wait-interpolation)

    (setq *grasp-state* "grasp-closing")
    (publish-grasp-state)
    (publish-info "grasping...")
    (let ((r-grasp-result (send *ri* :start-grasp *rarm*)))
	(send *ri* :wait-interpolation)
	(unless (< 20.0 r-grasp-result)
	  (publish-error "failed to grasp ~A" grasp-result)
	  (send *ri* :stop-grasp *rarm* :wait t)
	  (publish-info "back to pre grasp...")
	  (pr2-pregrasp-pose)
	  (return-from grasp-target-callback nil)))

    (setq *grasp-state* "picking")
    (publish-grasp-state)
    (publish-info "picking up...")
    (unix:sleep 10)
    (send *pr2* *rarm* :move-end-pos #f(0 0 150) :world)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
    (send *ri* :wait-interpolation)
    
    (judge-grasp-success)
    (publish-info "grasp succeeded!")


    (setq *grasp-state* "placing")
    (publish-grasp-state)
    (publish-info "placing ...")
    (send *pr2* *rarm* :move-end-pos #f(0 0 -150) :world)
    (send *ri* :wait-interpolation)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    (send *ri* :wait-interpolation)

    (setq *grasp-state* "grasp-opening")
    (publish-grasp-state)
    (send *ri* :stop-grasp *rarm* :wait t)
    (publish-info "return object...")

    (setq *grasp-state* "leaving")
    (publish-grasp-state)
    (send *pr2* *rarm* :move-end-pos #f(-160 0 0) :world)
    (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
    (send *ri* :wait-interpolation)
    (publish-info "pre grasp pose")

    (setq *grasp-state* "returning")
    (publish-grasp-state)
    (pr2-pregrasp-pose)
    (publish-info "click tabletop object!")
    )
  )

 

(defun wait-for-grasp-target ()
  (ros::subscribe *grasp-target-bounding-box-topic*
                  jsk_recognition_msgs::BoundingBox #'grasp-target-callback)
  (publish-info "click tabletop object!"))

(defun publish-grasp-state ()
  (ros::publish "grasp_state" (instance std_msgs::String :init :data *grasp-state*))) ;change from "/grasp_state" to "grasp_state"




