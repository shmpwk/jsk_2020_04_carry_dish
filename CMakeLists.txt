cmake_minimum_required(VERSION 3.0.2)
project(jsk_2020_4_carry_dish)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(catkin REQUIRED COMPONENTS
  catkin_virtualenv
  )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES jsk_2020_4_carry_dish
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

catkin_generate_virtualenv(
  PYTHON_INTERPRETER python2
  USE_SYSTEM_PACKAGES TRUE
  ISOLATE_REQUIREMENTS FALSE
  CHECK_VENV FALSE
  )

file(GLOB NODE_FILES node_scripts/*)
catkin_install_python(
  PROGRAMS ${NODE_FILES} 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(FILES requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )
