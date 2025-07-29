CPMAddPackage("gh:CLIUtils/CLI11@2.5.0")

add_executable(sample_minimal sample_minimal.cpp)
target_link_libraries(sample_minimal libcamera-opencv opencv_highgui)

add_executable(sample sample.cpp)
target_link_libraries(sample
    libcamera-opencv opencv_highgui CLI11::CLI11
)

add_executable(raw_fps raw_fps.cpp)
target_link_libraries(raw_fps
    libcamera-opencv PkgConfig::libcamera CLI11::CLI11)
