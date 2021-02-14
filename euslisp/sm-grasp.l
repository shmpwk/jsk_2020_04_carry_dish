#!/usr/bin/env roseus
(load "package://roseus_smach/src/state-machine.l")
(require "package://jsk_2020_4_carry_dish/euslisp/grasp_primitive.l")

(setq count 0)
;;(defun pr2-pregrasp-pose (&rest args)
;;  t)
;;(defun wait-for-grasp-target (&rest args)
;;  t)
(defun grasp-smach ()
  (let ((sm (instance state-machine :init)))
    (send sm :add-node (instance state :init :initialize 'pr2-pregrasp-pose))
    ;;(send sm :add-node (instance state :init :grasp-dish 'wait-for-grasp-target))
    ;;(send&rest args sm :add-node (instance state :init :grasp-dish 'popopo))
    (send sm :add-node (instance state :init :grasp-dish 'grasp-target-callback))
    (send sm :add-node (instance state :init :post-grasp 'postgrasp))
    ;; goal-states are generated in this method
    (send sm :goal-state (list :success :fail))

    ;; select a node as start-node
    (send sm :start-state :initialize)
    ;; from and to nodes are selected by name or symbol
    (send sm :add-transition :initialize :grasp-dish t)
    (send sm :add-transition :grasp-dish :post-grasp t)
    (send sm :add-transition :grasp-dish :fail nil)
    (send sm :add-transition :post-grasp :success t)
    (send sm :add-transition :post-grasp :fail nil)
    sm ))

