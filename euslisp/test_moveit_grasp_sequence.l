#!/usr/bin/env roseus
;; Author: Shumpei Wakabayashi <wakabayashi@jsk.imi.i.u-tokyo.ac.jp>
;; for all learning grasp system

(defvar *grasp-target-bounding-box-topic* "/bounding_box_marker/selected_box")
(defvar *grasp-status-topic* "/tabletop_object_grasp_status")
(defvar *base-frame-id* "/base_footprint")
(ros::load-ros-manifest "jsk_recognition_msgs")
(ros::load-ros-manifest "jsk_rviz_plugins")
(ros::roseus-add-msgs "geometry_msgs")
(require "models/arrow-object.l")


(ros::roseus "pr2_tabletop_object_grasp_node")
(ros::rate 10)

(require :pr2-interface "package://pr2eus/pr2-interface.l")
(load "package://pr2eus_moveit/euslisp/pr2eus-moveit.l")

(defun pr2-pregrasp-pose ()
  ;;(send *pr2* :reset-manip-pose)
  ;;(send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  ;;(send *ri* :wait-interpolation)
  ;;(send *pr2* :angle-vector #f(299.865 55.4297 53.3196 102.662 -121.542 125.071 -87.0418 -48.2131 -32.0168 -20.2598 -67.6931 -45.3044 -72.9084 -96.2568 -100.018 4.1047 54.75))
  ;;(send *pr2* :angle-vector #f(299.789 55.4297 53.3196 102.662 -121.541 125.071 -87.0426 -48.2131 -32.0168 -20.2592 -67.6931 -45.3044 -72.9084 -96.2581 -100.017 3.0787 60.2702))
  (send *pr2* :angle-vector #f(299.621 121.396 48.589 101.118 -121.542 132.021 -107.883 -36.5066 -31.575 -19.8811 -61.5465 -44.6159 -73.0576 -96.6719 -99.728 0.252701 63.078))
  ;;(send *pr2* :angle-vector #f(66.6328 55.4297 53.3196 102.662 -121.541 125.071 -87.0381 -48.2128 -32.0168 -20.2592 -67.6931 -45.3044 -72.9084 -96.2568 -100.02 3.3427 60.45))
  ;;(send *pr2* :angle-vector #f(66.5726 55.4297 53.3196 102.662 -121.542 125.071 -87.0381 -48.2128 -32.0168 -20.2598 -67.6931 -45.3044 -72.9051 -96.2568 -100.02 3.3487 56.826))

  (send *ri* :angle-vector (send *pr2* :angle-vector) 5000)
  (send *ri* :wait-interpolation)
  (setq *co-endeffector-vis* (arrow))
  (setq *co-ik-target* (arrow))
  (setq *pregrasp-point* (arrow))
  (setq *grasp-point* (arrow))
  (setq *box-arrow* (arrow))
  (objects (list *co-endeffector-vis* *co-ik-target* *pr2*))
  ;;(setq *co-target-vis* (arrow))
  ;;(objects (list *pr2* *co-endeffector-vis*))
  )

;; service callでdepthをcsvに落とす
(defun get-depth ()
  (ros::wait-for-service "/depth_data_collection/save_request")
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call "/depth_data_collection/save_request" req))
  )

(defun get-judge ()
  (ros::wait-for-service "/judge_data_collection/save_request")
  (setq req (instance std_srvs::TriggerRequest :init))
  (setq res (ros::service-call "/judge_data_collection/save_request" req))
  (print "judge data saved")
  )

;;; ;; sub-grasp-pointのcallback
;; (defun grasp-point-cb (msg)
;;   (print (list 'cb (send msg :pose :position :x)))
;;   (let ((cds (ros::tf-pose->coords (send msg :pose)))
;;         (base->parent (send *tfl* :lookup-transform
;;                             *base-frame-id* (send msg :header :frame_id)
;;                             (ros::time 0))))
;;     (send *tfb* :send-transform
;;           (send (send base->parent :copy-worldcoords)
;;                 :transform cds)
;;           *base-frame-id* "target_object" (ros::time 0)
;;           ))
;;      (send (send base->parent :copy-worldcoords) :transform cds))
;; )

;; sub-grasp-pointのcallback
(defun grasp-point-cb (msg)
  (print (list 'cb (send msg :pose :position :x)))
  (let ((cds (ros::tf-pose->coords (send msg :pose)))
        (base->parent (send *tfl* :lookup-transform
                            *base-frame-id* (send msg :header :frame_id)
                            (ros::time 0))))
     (send (send base->parent :copy-worldcoords) :transform cds)
     )
  )

;; one-shot-subscribe but not use now
;;(defun sub-grasp-point ()
;;  (let*(
;;         (msg (one-shot-subscribe "/grasp_point" geometry_msgs::PoseStamped
;;                                  ;;:after-stamp (ros::time 0)))
;;                                  ))
;;         (cds (when msg (ros::tf-pose->coords (send msg :pose))))
;;         (base->parent
;;           (when msg
;;               (send *tfl* :wait-for-transform *base-frame-id* (send msg :header :frame_id) (ros::time 0) 0.1)
;;               (send *tfl* :lookup-transform
;;                           *base-frame-id* (send msg :header :frame_id)
;;                           (ros::time 0)))))
;;    ;;(print (rpy-angle (send target-point :worldrot)))
;;    (setq *cds* cds)
;;    (when (and cds base->parent)
;;      ;;(send *tfb* :send-transform
;;        (send (send base->parent :copy-worldcoords) :transform cds))
;;      ;;      *base-frame-id* "target_object" (send msg :header :stamp))
;;    )
;;  )
;;
;;;;(ros::subscribe "grasp_point" geometry_msgs::posestamped #'grasp-point-cb)
;;
(defun tf-grasp-point (msg header)
    (let*(
         (cds (when msg (ros::tf-pose->coords msg)))
         (base->parent
           (when msg
               (send *tfl* :wait-for-transform *base-frame-id* (send header :frame_id) (ros::time 0) 0.1)
               (send *tfl* :lookup-transform
                           *base-frame-id* (send header :frame_id)
                           (ros::time 0)))))
    (setq *cds* cds)
    (when (and cds base->parent)
      ;;(send *tfb* :send-transform
        (send (send base->parent :copy-worldcoords) :transform cds))
      ;;      *base-frame-id* "target_object" (send msg :header :stamp))
    )
  )

;; 対象物を持ち上げられたか否か判断する
(defun judge-grasp-success ()
  ;;service call
  )

(defun publish-status (msg-type fmt)
  (let ((color
         (case msg-type
          ('error (instance std_msgs::ColorRGBA :init
                            :r 1.0 :g 0.0 :b 0.0 :a 1.0))
          ('warn (instance std_msgs::ColorRGBA :init
                           :r 1.0 :g 1.0 :b 0.0 :a 1.0))
          (t (instance std_msgs::ColorRGBA :init
                       :r (/ 157.0 255) :g (/ 204.0 255) :b (/ 224.0 255) :a 1.0)))))
    (unless (ros::get-topic-publisher *grasp-status-topic*)
      (ros::advertise *grasp-status-topic* jsk_rviz_plugins::OverlayText 5)
      (unix:sleep 1))
    (ros::publish *grasp-status-topic*
                  (instance jsk_rviz_plugins::OverlayText :init
                            :width 1920 :height 400
                            :left 0 :top 0
                            :line_width 10 :text_size 50
                            :fg_color color
                            :text fmt))))

(defun publish-info (&rest args)
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-info s)
    (publish-status 'info s)))
(defun publish-warn (&rest args)
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-warn s)
    (publish-status 'warn s)))
(defun publish-error (&rest args)
  (let ((s (format nil (car args) (cdr args))))
    (ros::ros-error s)
    (publish-status 'error s)))

(defun vector3-> (applier v)
  (funcall applier
         (* (send v :x) 1000.0)
         (* (send v :y) 1000.0)
         (* (send v :z) 1000.0)))

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
    (setq *box* (send (apply #'make-cube d)
          :transform
          (send (send base->parent :copy-worldcoords)
                :transform cds))
                )
    (send *box-arrow* :newcoords (send *box* :copy-worldcoords))
    )
  )

(defun grasp-target-callback (msg)
  (print "clicked")
  (setq msgarr (one-shot-subscribe "/grasp_points" geometry_msgs::PoseArray))
  (unless msgarr
    (ros::ros-logerr "target point is nil !!")
    (return-from grasp-target-callback nil))
  (print "grasp_points subscribed!!!")
  ;;(mapcar #'(lambda (msg) (send msgarr :poses)))
  (dolist (point (send msgarr :poses))
    (setq target-point (tf-grasp-point point (send msgarr :header)))
    (unless target-point
      (ros::ros-logerr "target point is nil !!")
      (return-from grasp-target-callback nil))
    ;; gradient from grasp point to the middle of the box
    (setq grad (v- (send (bounding-box->cube msg) :worldpos)
                            (send target-point :worldpos)))
    ;; grasping toward centripetal direction
    ;;(setq *rot* (- 1.57 (atan2  (elt grad 1) (elt grad 0))))
    (setq *rot*  (atan2  (elt grad 1) (elt grad 0)))
    ;; pre;;grasp pos
    (setq *pre-pos* (float-vector (elt (scale 3 grad) 0) (elt (scale 3 grad) 1) 60))
    (setq *reach-pos* (float-vector (elt (scale -1.5 grad) 0) (elt (scale -1.5 grad) 1) -100))
    ;; pre grasp rotation
    ;;(setq rdm (+ (/ pi 2) (elt (elt (rpy-angle (send *cds* :rot)) 0) 1)))
    (setq rdm (elt (elt (rpy-angle (send *cds* :rot)) 1) 1))
    (setq *grasp-rot* (float-vector *rot* rdm -1.54))
    (print "actual grasp point")
    (print *grasp-rot*)
  
    (if *grasping-object-p*
        (progn
          (publish-warn "cancel grasping...")
          (send *ri* :stop-motion)
          (send *ri* :stop-grasp *rarm* :wait t)
          (setq *grasping-object-p* nil))
      (setq *grasping-object-p* t))
    (setq reachable nil)
    (when
      (send *pr2* *rarm* :inverse-kinematics
            ;;(make-coords :pos target-point :worldpos)
            (make-coords :pos (v+ (send target-point :worldpos)
                                  *pre-pos*)
                                  ;;(float-vector -100 0 50))
                         ;;:rpy (float-vector 0 1.2 -1.54)) ;;success(0 1.2 -1.54)
                         :rpy *grasp-rot*))
      (send *pregrasp-point* :newcoords
            (make-coords :pos (v+ (send target-point :worldpos)
                                    ;;(float-vector -100 0 150))
                                    *pre-pos*)
                           ;;:rpy (float-vector 0 1.2 -1.54)))
                           :rpy *grasp-rot*))
      (send *co-ik-target* :newcoords (send target-point :copy-worldcoords))
      (objects (list *pr2* *box* *co-ik-target* *pregrasp-point* *box-arrow*))
      (publish-info "reachable")
      (setq reachable t)
      (return nil)))
  (unless reachable
    (publish-error "out of reach...")
    (send *pregrasp-point* :newcoords
          (make-coords :pos (v+ (send target-point :worldpos)
                                  *pre-pos*)
                                  ;;(float-vector -100 0 150))
                         ;;:rpy (float-vector -1.57 1.2 -1.54)))
                         :rpy *grasp-rot*))
    (send *co-ik-target* :newcoords (send target-point :copy-worldcoords))
    (objects (list *pr2* *box* *co-ik-target* *pregrasp-point* *box-arrow*))
    (setq *grasp-state* "out of reach")      
    (publish-grasp-state)
    (publish-info "grasp failed!")          
    (return-from grasp-target-callback nil))
  ;;(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  (send *ri* :angle-vector-motion-plan (send *pr2* :angle-vector) :total-time 3000 :ctype :rarm-controller :start-offset-time 0 :move-arm :rarm) ;;for moveit
  (send *ri* :wait-interpolation) 

  (setq *grasp-state* "reaching")
  (publish-grasp-state)
  ;;(publish-info "pre grasp pose...")
  (send *ri* :stop-grasp *rarm* :wait t)
  ;;(send *pr2* :head :look-at-hand *rarm*)
  ;;(send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
  ;;(send *ri* :angle-vector-motion-plan (send *pr2* :angle-vector) :total-time 1000 :ctype :rarm-controller :start-offset-time 0 :move-arm :rarm) ;;for moveit
  ;;(send *ri* :wait-interpolation)

  (send *co-endeffector-vis* :newcoords (send *pr2* :copy-worldcoords))
  (send *grasp-point* :newcoords 
        (make-coords :pos (v+ (send target-point :worldpos)
                              (float-vector 0 0 -30)) ;;(-200+105, 0+0, 150-150)
                     :rpy (float-vector 0 1.2 -1.54)))
  (objects (list *pr2* *box* *co-endeffector-vis* *co-ik-target* *grasp-point* *box-arrow*))
  
  (setq *grasp-state* "grasp-approaching")
  (publish-grasp-state)
  (publish-info "reaching...")
  (send *pr2* *rarm* :move-end-pos *reach-pos* :world :look-at-target t)
  ;;(send *pr2* :head :look-at-hand *rarm*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)

  (setq *grasp-state* "grasp-closing")
  (publish-grasp-state)
  (publish-info "grasping...")
  (let ((grasp-result (send *ri* :start-grasp *rarm* :gain 0.1)))
    (send *ri* :wait-interpolation)
    (unix:sleep 3)
    (print grasp-result)
    (unless (< 4.5 grasp-result)
      (publish-error "failed to grasp ~A" grasp-result)
      (send *ri* :stop-grasp *rarm* :wait t)
      (publish-info "back to pre grasp...")
      (pr2-pregrasp-pose)
      (return-from grasp-target-callback nil)))

  (setq *grasp-state* "picking")
  (publish-grasp-state)
  (publish-info "picking up...")
  ;;(unix:sleep 1)
  (send *pr2* *rarm* :move-end-pos #f(0 0 270) :world)
  ;;(send *pr2* :head :look-at-hand *rarm*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (let ((grasp-result (send *ri* :start-grasp *rarm*)))
    (print grasp-result)
    (if (< 7 grasp-result)   ;;for gazebo (< 0.1 grasp-result)                          
      (progn                                                     
        (setq *grasp-state* "grasp succeeded!")               
        (publish-grasp-state)
        (publish-info "grasp succeeded!")                   
        (ros::set-param (format nil "~/judge_grasp_success") 1)
        ;;(send *ri* :go-pos-unsafe 0 -0.65 0)
        (send *ri* :go-pos-unsafe -0.3 0 0)
        (send *ri* :go-pos-unsafe 0 0 -90)
        (send *ri* :go-pos-unsafe 0.4 0 0)
        (setq *grasp-state* "placing")
        (publish-grasp-state)
        (publish-info "placing ...")
        (send *pr2* *rarm* :move-end-pos #f(100 0 -240) :world)
        (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
        (send *ri* :wait-interpolation)
        ;; open glipper
        (send *ri* :stop-grasp *rarm* :wait t)
        (send *ri* :angle-vector (send *pr2* :angle-vector) 3000)
        (send *ri* :wait-interpolation)
        (send *pr2* *rarm* :move-end-pos #f(-100 0 100) :world)
        (send *ri* :angle-vector (send *pr2* :angle-vector) 2000)
        (publish-info "pre grasp pose")
        (setq *grasp-state* "returning")
        (pr2-pregrasp-pose)
        ;;(send *ri* :go-pos-unsafe 0 0.65 0)
        (send *ri* :go-pos-unsafe -0.4 0 0)
        (send *ri* :go-pos-unsafe 0 0 90)
        (send *ri* :go-pos-unsafe 0.3 0 0)
        (send *ri* :wait-interpolation)
        )
      (progn                                          
        (setq *grasp-state* "grasp failed!")      
        (publish-grasp-state)
        (publish-info "grasp failed!")          
        (ros::set-param (format nil "~/judge_grasp_success") 0) 
        (pr2-pregrasp-pose))))

  (publish-grasp-state)
  (publish-info "click tabletop object!")
  )

(defun wait-for-grasp-target ()
  (ros::subscribe *grasp-target-bounding-box-topic*
                  jsk_recognition_msgs::BoundingBox #'grasp-target-callback)
  (publish-info "click tabletop object!"))

(defun publish-grasp-state ()
  (ros::publish "grasp_state" (instance std_msgs::String :init :data *grasp-state*))) ;change from "/grasp_state" to "grasp_state"

(defun demo ()
  (setq *grasping-object-p* nil)
  (setq *rarm* :rarm)
  (setq *tfl* (instance ros::transform-listener :init))
  (setq *tfb* (instance ros::transform-broadcaster :init))
  (ros::advertise "/grasp_state" std_msgs::String 1)
  (setq *grasp-state* "waiting")
  (pr2-init)
  (pr2-pregrasp-pose)
  (wait-for-grasp-target)
  (ros::spin))
(demo)

