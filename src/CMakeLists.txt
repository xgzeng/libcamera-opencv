
add_library(libcamera-opencv
    libcamera_cv.hpp
    libcamera_cv.cpp
    camera_capture_impl.hpp
    camera_capture_impl.cpp
)

target_include_directories(libcamera-opencv
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE
    ${LIBCAMERA_INCLUDE_DIRS}
)

target_link_libraries(libcamera-opencv
  PUBLIC
    ${OpenCV_LIBS}
  PRIVATE
    ${LIBCAMERA_LIBRARIES})

target_compile_options(libcamera-opencv
  PRIVATE
    ${LIBCAMERA_CFLAGS_OTHER})
