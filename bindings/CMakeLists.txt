find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(eigenpy 3.9.1 REQUIRED)

set(
  SIMPLE_MPC_PYTHON_SOURCES
  module.cpp
  expose-robot-handler.cpp
  expose-problem.cpp
  expose-mpc.cpp
  expose-interpolate.cpp
  expose-centroidal.cpp
  expose-fulldynamics.cpp
  expose-kinodynamics.cpp
  expose-inverse-dynamics.cpp
  expose-friction-compensation.cpp
)

Python3_add_library(simple_mpc_pywrap MODULE WITH_SOABI ${SIMPLE_MPC_PYTHON_SOURCES})
target_link_libraries(simple_mpc_pywrap
  PRIVATE
  simple-mpc
  eigenpy::eigenpy
  Python3::Module
)
set_target_properties(simple_mpc_pywrap PROPERTIES PREFIX "")

set(SIMPLE_MPC_PYTHON_DIR "${PROJECT_SOURCE_DIR}/bindings/simple_mpc")
add_custom_command(
  TARGET simple_mpc_pywrap
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:simple_mpc_pywrap>
          "${SIMPLE_MPC_PYTHON_DIR}/$<TARGET_FILE_NAME:simple_mpc_pywrap>"
  COMMENT "Copying simple_mpc_pywrap to bindings/simple_mpc/"
)
