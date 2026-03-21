find_package(Boost REQUIRED COMPONENTS unit_test_framework)
find_package(example-robot-data 4.0.9 REQUIRED)

set(
  SIMPLE_MPC_TESTS
  robot_handler
  problem
  inverse-dynamics/kinodynamics-id
  inverse-dynamics/centroidal-id
  friction
  interpolator
)

function(add_simple_mpc_test filename)
  string(REPLACE "/" "_" test_slug "${filename}")
  set(test_target "test-cpp-${test_slug}")
  set(test_module "${test_slug}Test")
  string(REPLACE "-" "_" test_module "${test_module}")

  add_executable(${test_target} "${filename}.cpp")
  target_include_directories(${test_target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(
    ${test_target}
    PRIVATE
      BOOST_TEST_DYN_LINK
      BOOST_TEST_MODULE=${test_module}
  )
  target_link_libraries(
    ${test_target}
    PRIVATE
      simple-mpc
      example-robot-data::example-robot-data
      Boost::unit_test_framework
  )

  add_test(NAME ${test_target} COMMAND ${test_target})
endfunction()

foreach(test_name IN LISTS SIMPLE_MPC_TESTS)
  add_simple_mpc_test(${test_name})
endforeach()
