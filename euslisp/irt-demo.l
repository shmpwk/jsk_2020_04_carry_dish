#!/usr/bin/env roseus

;; load robot, room and arrow
(load "package://pr2eus/pr2.l")   
(load "models/room73b2-scene.l")
(load "models/arrow-object.l")

(defun init()
  ;;set robot
  (setq *robot* (pr2))
  ;; set room73b2
  (setq *room73b2* (room73b2))
  ;; set arrow
  (setq *axis* (arrow))
  ;; IRT viewer
  (objects (list *robot* *room73b2* *axis*))
  ;; set cup
  (setq *cup* (send *room73b2* :object "room73b2-mug-cup"))
  (if (send *cup* :parent) (send (send *cup* :parent) :dissoc *cup*))
  ;; set dish
  (setq *dish* (send *room73b2* :object "room73b2-dish"))
  (if (send *dish* :parent) (send (send *dish* :parent) :dissoc *dish*))

  (format t ";; please run (main)~%")
  )

;;シンクの前に移動
(defun go-to-sink ()
  (send *robot* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")  :world)
  (send *robot* :move-to (make-coords :pos #f(300 -150 0))))

;;皿の前に移動
(defun go-to-dish ()
  (send *robot* :head :look-at (send *dish* :worldpos))
  )

;;皿に手を伸ばす
(defun reach-dish ()
  (send *axis* :newcoords (send *dish* :copy-worldcoords))
  (send *axis* :transform (car (send *dish* :handle)))
  (setq c (car (send *dish* :handle)))
  (send *robot* :rarm :inverse-kinematics c
	:link-list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
	:move-target (send *robot* :rarm :end-coords)
	:translation-axis t
	:rotation-axis nil
	:revert-if-fail nil)
  (x::window-main-one)
  (send *irtviewer* :draw-objects :flush nil)
  (send c :draw-on :flush t)
  (unix:sleep 1)
  )
)

;;皿を掴みたかった（掴めてない）
(defun grasp-dish ()
  (send (send *robot* :rarm :end-coords :parent) :assoc *dish*)
  (send (send *robot* :rarm :end-coords :parent) :assoc *axis*)
  (send *robot* :rarm :move-end-pos #f(-400 -350 -200)
	:local
	:rotation-axis t
	:revert-if-fail nil))
  (unix:sleep 1)
)


;;食器棚の前に移動
(defun go-to-cupboard ()
  (send *robot* :move-to  (send (room73b2-cupboard-left) :transform (make-coords :pos (float-vector 2048.0 (- 3526.5 550) 0.0) :rot #2f((2.220446e-16 -1.0 0.0) (1.0 2.220446e-16 0.0) (0.0 0.0 1.0)))) :world)
  (unix:sleep 1)
  )

;;皿を置く
(defun put-dish ()
  (send (send *robot* :rarm :end-coords :parent) :dissoc *dish*)
  (send (send *robot* :rarm :end-coords :parent) :dissoc *axis*)
  )

;;*irtviewer*上のロボットを拡大表示
(defun view-adj ()
  (send *irtviewer* :look-all
	(geo::make-bounding-box
	 (flatten (send-all (send *robot* :bodies) :vertices))))
  )

(defun main ()
  (init)
  (go-to-sink)
  (go-to-dish)
  (reach-dish)
  (grasp-dish)
  (go-to-cupboard)
  (put-dish)
  (go-to-sink)
  (view-adj)
  )

(main)
