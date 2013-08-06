add_executable(testPolarCalibration 
    polarcalibration.cpp 
    main.cpp
    calibration.pb.cc
)
  
include_directories(testPolarCalibration
#     GPULibSVM
#     ${EIGEN3_INCLUDE_DIR}
#     ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
#     emon.a
#     ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(testPolarCalibration
#   ${EIGEN3_LIBRARIES}
#   ${PCL_LIBRARIES}
  ${OpenCV_LIBS}
#   ${CUDA_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PROTOBUF_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
#   ${Boost_
#   ${CGAL_CGAL_LIBRARY}
#   ${GMP_LIBRARIES}
#   ${CUDA_CURAND_LIBRARIES}
#   emon
)
