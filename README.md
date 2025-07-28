# LibCamera OpenCV Integration

A C++ library that provides OpenCV VideoCapture compatible(best effort) interface for libcamera, enabling easy camera capture on systems that use libcamera (like Raspberry Pi).

## Requirements

- libcamera (>=0.5)
- OpenCV 4.x
- CMake 3.10+
- C++17 compatible compiler
- CLI11 library (required by sample application)

### Tested Environment:
* Raspberry Pi 5
* Ubuntu 24.04 (with libcamera 0.5 backported)
* IMX477 Camera (x2)

## Building

```bash
cmake --preset <debug|release>
cmake --build --preset <debug|release>
```

## Usage

### Basic Camera Capture

```cpp
#include <libcamera_cv.hpp>
#include <opencv2/highgui.hpp>

int main() {
    cv::CameraCapture cap(1);
    cv::Mat frame;
    while (cap.read(frame)) {
        cv::imshow("Camera", frame);
        if (cv::pollKey() == 'q') break;
    }
    return 0;
}
```

### Sample Application

The included sample application demonstrates multi-camera capture with frame merging:

```bash
# Single camera capture
./build/debug/bin/sample

# Set resolution and frame rate
./build/debug/bin/sample -w 1280 -h 720 -f 30

# Capture for 10 seconds
./build/debug/bin/sample -t 10

# Disable display window
./build/debug/bin/sample --no-window

# Enable camera flips
./build/debug/bin/sample --hflip --vflip

# Multiple camera capture with merged view
./build/debug/bin/sample -c 1 -c 2
```

## API Reference

See OpenCV::VideoCapture documentation for details on the API.

### Supported Properties

- `cv::CAP_PROP_FRAME_WIDTH`: Frame width
- `cv::CAP_PROP_FRAME_HEIGHT`: Frame height
- `cv::CAP_PROP_FOURCC`: Pixel format (FourCC code)
- `cv::CAP_PROP_FPS`: Frame rate

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
