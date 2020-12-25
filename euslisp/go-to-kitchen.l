#!/usr/bin/env roseus

(require :pr2-interface "package://pr2eus/pr2-interface.l")

;; load scene for kinematics simulater
(require :pr2-move "package://jsk_demo_common/euslisp/pr2-move.l")

(load "models/room73b2-scene.l")
(load "models/white-bread-salad-dish-object")

(defun go-to-sink ()
  (send *pr2* :move-to (send *room73b2* :spot "cook-spot") :world)
  (send *ri* :go-pos 1.1 0 90)
  ;;(go-to-spot :wait t :spot-name "cook-spot")
  (send *pr2* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")  :world)
  (send *pr2* :move-to (make-coords :pos #f(300 -150 0)))
  (send *ri* :move-to (send *room73b2* :spot "/eng2/7f/room73B2-sink-front")) ;;why not :world?
  (send *ri* :go-pos 0.3 -0.15 0) 
)
(pr2-init)    
(setq *room73b2* (room73b2))
(go-to-sink)
