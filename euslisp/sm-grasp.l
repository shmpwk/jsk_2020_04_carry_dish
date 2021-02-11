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
    (send sm :add-node (instance state :init :initial 'pr2-pregrasp-pose))
    ;;(send sm :add-node (instance state :init :grasp-dish 'wait-for-grasp-target))
    (send sm :add-node (instance state :init :grasp-dish 'popopo))
    ;; goal-states are generated in this method
    (send sm :goal-state (list :success :fail))

    ;; select a node as start-node
    (send sm :start-state :initial)
    ;; from and to nodes are selected by name or symbol
    (send sm :add-transition :initial :grasp-dish t)
    (send sm :add-transition :grasp-dish :success t)
    (send sm :add-transition :grasp-dish :fail nil)
    sm ))

