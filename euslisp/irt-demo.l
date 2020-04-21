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
  (setq *dish* (send *room73b2* :objsct "room73b2-dish"))
  (if (send *dish* :parent) (send (send *dish* :parent) :dissoc *dish*))

  (format t ";; please run (main)~%")
  )

;;シンクに移動
(defun go-to-sink ()
  (send *robot* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")  :world)
  (send *robot* :move-to (make-coords :pos #f(300 -150 0))))

;;*irtviewer*上のロボットを拡大表示
(defun view-adj ()
  (send *irtviewer* :look-all
	(geo::make-bounding-box
	 (flatten (send-all (send *robot* :bodies) :vertices))))
  )

(defun main ()
  (init)
  (go-to-sink)
  (view-adj)
  )

(main)
