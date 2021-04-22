#! /usr/bin/env roseus 

(ros::load-ros-manifest "actionlib_msgs")
(ros::load-ros-manifest "behavior_tree_core")
(ros::roseus "pr2_tabletop_object_grasp_node")
                        
(defun set-status (status)
  (setq *feedback* (instance behavior_tree_core::BTFeedback))
  (setq *result* (instance behavior_tree_core::BTResult))         
  (send *result* :status (send *feedback* :status))
  (setd *req* :set-succeeded (send *result*))
  (print "set status")
)                       
                                                      
(defun judge-cb (server goal)
  (print "cb start")         
  (set-status (goal))    
  (unix:sleep 1)
  (print "cb end")
)                                                                                             
(defun get-judge ()                                                              
  ;;(ros::wait-for-service "action")
  (setq *req* (instance ros::simple-action-server :init "grasp" behavior_tree_core::BTAction :execute-cb 'judge-cb))
  (ros::rate 10)                                                           
  (print "judged data save")                                          
) 

(get-judge)                                                                                         
