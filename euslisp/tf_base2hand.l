#!/usr/bin/env roseus

(ros::roseus "tf_broadcaster")

(setq *b* (instance ros::transform-broadcaster :init))
(do-until-key
  (setq tm (ros::time-now))
  (send *b* :send-transform (make-coords :pos #f(
