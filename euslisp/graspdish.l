#!/usr/bin/env roseus
;; -*- coding: utf-8 -*-

(load "package://pr2eus/pr2-interface.l")
(load "models/kettle-object.l")
(load "models/white-bread-salad-dish-object")
(load "models/arrow-object.l")


(setq *ri* (instance pr2-interface :init))
(setq *robot* (pr2))
(send *robot* :init) 
(setq *kettle* (kettle))
(setq *dish* (white-bread-salad-dish))
(send *kettle* :locate #f(500 0 500))
(send *dish* :locate #f(400 0 500))
(setq *axis* (arrow))
(setq *co-endeffector* (arrow))
(setq *co-ik-target* (arrow))
;;(setq c (car (send *kettle* :handle)))
(setq c (car (cdr (cdr(send *dish* :handle)))))
;;(setq *target-coords* (send (send *kettle* :worldcoords) :copy-worldcoords))
(setq *target-coords* (send (send *dish* :worldcoords) :copy-worldcoords))
(send *co-endeffector* :newcoords (send (send *kettle* :worldcoords) :copy-worldcoords))
(send *co-ik-target* :newcoords (send c :copy-worldcoords))

;;(setq *target-coords* (send (send *dish* :worldcoords) :copy-worldcoords))
;;(objects (list *robot* *dish* *target-coords* *axis*))
(objects (list *robot* *dish* *axis* *co-endeffector* *co-ik-target*))
(send *ri* :start-grasp :rarm)
(send *robot* :inverse-kinematics c
      :link-list (send *robot* :link-list (send *robot* :rarm :end-coords :parent))
      :move-target (send *robot* :rarm :end-coords)
      :rotation-axis :nil)
