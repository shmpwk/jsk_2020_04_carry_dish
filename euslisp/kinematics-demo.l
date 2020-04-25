#!/usr/bin/env roseus

(require :pr2-interface "package://pr2eus/pr2-interface.l")

;; load scene for kinematics simulater
(require :pr2-move "package://jsk_demo_common/euslisp/pr2-move.l")

(load "models/room73b2-scene.l")
(load "models/white-bread-salad-dish-object")
(load "models/arrow-object.l")

(defun init()
  ;;set robot
  (setq *ri* (instance pr2-interface :init))
  (setq *robot* (pr2))
  (send *robot* :init) 
  ;; set room73b2
  (setq *room73b2* (room73b2))
  ;; set arrow
  (setq *axis* (arrow))
  ;; IRT viewer
  (objects (list *robot* *room73b2* *axis*))
  ;; set dish
  (setq *dish* (send *room73b2* :object "room73b2-dish"))
  (if (send *dish* :parent) (send (send *dish* :parent) :dissoc *dish*))
  ;; set 73b2
  (if (send *ri* :simulation-modep)
      (progn
	(send *ri* :objects (send *room73b2* :objects))
	))
  ;; IRT viewer
  (objects (list *robot* *room73b2* *axis*))
  )

;;シンクの前に移動
(defun go-to-sink ()
  (send *robot* :move-to (send *room73b2* :spot "cook-spot") :world)
  (send *ri* :go-pos 1.1 0 90)
  ;;(go-to-spot :wait t :spot-name "cook-spot")
  (send *robot* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")  :world)
  (send *robot* :move-to (make-coords :pos #f(300 -150 0)))
  (send *ri* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")) ;;why not :world?
  (send *ri* :go-pos 0.3 -0.15 0) 
)

;;皿に手を伸ばす
(defun reach-dish ()
  (send *robot* :head :look-at (send *dish* :worldpos))
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  ;;(send *axis* :newcoords (send *dish* :copy-worldcoords))
  ;;(send *axis* :transform (car (send *dish* :handle)))
  ;;(send *axis* :rotate (deg2rad -90) :y :world)
  (setq *c* (car (cdr (cdr (cdr (cdr (send *dish* :handle)))))))
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 50) :world)
  (send *robot* :rarm :move-end-pos #f(-20 0 100) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  (send *robot* :rarm :inverse-kinematics *c*
	:rotation-axis t
	)
  (send *ri* :angle-vector (send *robot* :angle-vector) 2000)
  (send *ri* :wait-interpolation)
  )
;;皿を掴む
(defun grasp-dish ()
  (send (send *robot* :rarm :end-coords :parent) :assoc *dish*)
  ;;(send (send *robot* :rarm :end-coords :parent) :assoc *axis*)

  ;;(setq *target-coords* (send (send *dish* :worldcoords) :copy-worldcoords))
  (send *ri* :start-grasp :rarm :wait t)
  ;;(send *robot* :inverse-kinematics c
	;;:link-list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
	;;:move-target (send *robot* :rarm :end-coords)
	;;:rotation-axis :nil)
  (unix:sleep 1)
  (send *robot* :rarm :move-end-pos #f(0 0 150) :world)
  ;;(send *robot* :rarm :move-end-pos #f(0 0 100) :world)
  ;;(send *robot* :rarm :move-end-pos #f(0 0 100) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (send *robot* :rarm :move-end-pos #f(100 -200 0) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (send *robot* :rarm :move-end-pos #f(100 0 0) :world)
  (send *robot* :rarm :move-end-pos #f(-100 -100 0) :world)
  (send *robot* :rarm :move-end-pos #f(-50 -100 0) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (setq *sink-back* (make-coords :pos #f(-300 0 0) :rpy #f(0 0 0)) )
  (send *robot* :move-to *sink-back* :local)
  (send *ri* :go-pos -0.3 0 0)
  (send *robot* :rarm :move-end-pos #f(0 -150 -50) :world)
  (send *robot* :rarm :move-end-pos #f(0 -150 0) :world)
  (send *robot* :rarm :move-end-pos #f(0 -150 0) :world)
  (send *robot* :rarm :move-end-pos #f(50 0 0) :world)
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  )

;;食器棚の前に移動
(defun go-to-cupboard ()
  (
  (setq *turn-right* (make-coords :pos #f(0 0 0) :rpy #f(-1.57 0 0)))
  (send *robot* :move-to *turn-right* :local)
  (send *ri* :go-pos 0 0 -1.57)
  (send *robot* :move-to  (send (room73b2-cupboard-left) :transform (make-coords :pos (float-vector 2048.0 (- 3526.5 550) 0.0) :rot #2f((2.220446e-16 -1.0 0.0) (1.0 2.220446e-16 0.0) (0.0 0.0 1.0)))) :world)
  (send *ri* :move-to  (send (room73b2-cupboard-left) :transform (make-coords :pos (float-vector 2048.0 (- 3526.5 550) 0.0) :rot #2f((2.220446e-16 -1.0 0.0) (1.0 2.220446e-16 0.0) (0.0 0.0 1.0)))))
  )
  
;;皿を置く
(defun put-dish ()
  (send (send *robot* :rarm :end-coords :parent) :dissoc *dish*)
  (send (send *robot* :rarm :end-coords :parent) :dissoc *axis*)
  (send *ri* :stop-grasp :larm :wait t)
  (send *robot* :larm :move-end-pos #f(40 0 0) :world
	:debug-view t :look-at-target t)
  (send *ri* :angle-vector (send *robot* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  )



;;*irtviewer*上のロボットを拡大表示
(defun view-adj ()
  (send *irtviewer* :look-all
	(geo::make-bounding-box
	 (flatten (send-all (send *robot* :bodies) :vertices)))))

(defun main ()
  (init)
  (go-to-sink)
  (reach-dish)
  (grasp-dish)
  (go-to-cupboard)
  ;(put-dish)
  ;(go-to-sink)
  ;(view-adj)
  )
