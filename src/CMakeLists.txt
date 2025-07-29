
add_library(libcamera-opencv
    libcamera_cv.hpp
    libcamera_cv.cpp
    camera_capture_impl.hpp
    camera_capture_impl.cpp
)

target_include_directories(libcamera-opencv
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(libcamera-opencv
  PUBLIC
    opencv_core
  PRIVATE
    opencv_imgproc
    PkgConfig::libcamera
)
