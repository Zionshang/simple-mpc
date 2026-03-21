find_package(benchmark REQUIRED)
find_package(example-robot-data 4.0.9 REQUIRED)

add_custom_target(run-benchmarks COMMAND ${CMAKE_COMMAND} -E echo "All benchmarks completed.")

function(add_simple_mpc_benchmark source_file)
  get_filename_component(benchmark_name "${source_file}" NAME_WE)
  set(target_name "bench-${benchmark_name}")

  add_executable(${target_name} "${source_file}")
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(
    ${target_name}
    PRIVATE
      simple-mpc
      example-robot-data::example-robot-data
      benchmark::benchmark
  )

  add_custom_target(run-${target_name}
    COMMAND ${target_name}
    DEPENDS ${target_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  add_dependencies(run-benchmarks run-${target_name})
endfunction()

add_simple_mpc_benchmark(talos.cpp)
add_simple_mpc_benchmark(go2.cpp)
