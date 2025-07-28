CPMAddPackage("gh:CLIUtils/CLI11@2.5.0")

add_executable(sample_minimal sample_minimal.cpp)
target_link_libraries(sample_minimal libcamera-opencv ${OpenCV_LIBS})

add_executable(sample sample.cpp)
target_link_libraries(sample
  libcamera-opencv ${OpenCV_LIBS}
  CLI11::CLI11
)
