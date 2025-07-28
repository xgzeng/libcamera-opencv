#include <libcamera_cv.hpp>
#include <chrono>
#include <iomanip>

#include <opencv2/core/utils/logger.hpp>
namespace cv_logging = cv::utils::logging;

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <optional>
#include <CLI/CLI.hpp>

#include <csignal>
#include <atomic>

std::atomic<bool> stop_flag {false};

void handle_sigint(int)
{
    stop_flag = true;
}

// CV_FOURCC from new version of <opencv2/core/cvdef.h>
// #include <opencv2/core/cvdef.h>
/** @brief Constructs the 'fourcc' code, used in video codecs and many other
   places. Simply call it with 4 chars like `CV_FOURCC('I', 'Y', 'U', 'V')`
*/
static constexpr int CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

constexpr int BG24 = CV_FOURCC('B', 'G', '2', '4');
constexpr int RG24 = CV_FOURCC('R', 'G', '2', '4');
// greyscale
constexpr int R8 = CV_FOURCC('R', '8', ' ', ' ');
constexpr int R16 = CV_FOURCC('R', '1', '6', ' ');

// Helper function to decode FourCC integer to string
static std::string fourccToString(int fourcc)
{
    char code[5] = {0};
    code[0] = (fourcc & 0xFF);
    code[1] = ((fourcc >> 8) & 0xFF);
    code[2] = ((fourcc >> 16) & 0xFF);
    code[3] = ((fourcc >> 24) & 0xFF);
    return std::string(code);
}

// Structure to hold camera configuration options
struct CameraConfig
{
    std::optional<int> width;
    std::optional<int> height;
    std::optional<double> frame_rate;
    bool hflip = false;
    bool vflip = false;
};

// Structure to hold application options
struct AppOptions
{
    std::vector<int> camera_indices = {1};  // Default to camera 1
    int timeout = 0;  // capture duration(seconds), 0 = infinite
    bool show_window = true;
    bool verbose = true;
};

// Function to configure a single camera with the given options
bool configureCameraCapture(cv::CameraCapture& cap, int camera_index, const CameraConfig& config)
{
    if (!cap.open(camera_index))
    {
        CV_LOG_ERROR(NULL, "Could not open camera " << camera_index);
        return false;
    }

    CV_LOG_INFO(NULL, "Camera " << camera_index << " opened successfully");

    // print default camera properties
    CV_LOG_INFO(NULL, "Camera " << camera_index << " default properties:");
    CV_LOG_INFO(NULL, "  Frame Width: " << cap.get(cv::CAP_PROP_FRAME_WIDTH));
    CV_LOG_INFO(NULL, "  Frame Height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    CV_LOG_INFO(NULL,
                "  FourCC: " << fourccToString(static_cast<int>(cap.get(cv::CAP_PROP_FOURCC))));
    CV_LOG_INFO(NULL, "  FPS: " << cap.get(cv::CAP_PROP_FPS));

    if (config.width)
    {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, config.width.value());
    }
    if (config.height)
    {
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, config.height.value());
    }
    if (config.frame_rate)
    {
        cap.set(cv::CAP_PROP_FPS, config.frame_rate.value());
    }

    // Apply flip settings for libcamera-based capture
#ifndef USE_VIDEOCAPTURE
    cap.setOrientation(config.hflip, config.vflip);
#endif

    return true;
}

// Function to merge multiple frames into a single composite frame
cv::Mat mergeFrames(const std::vector<cv::Mat>& frames,
                    const std::vector<int>& camera_indices,
                    double fps = 0.0)
{
    if (frames.empty())
        return cv::Mat();

    // Calculate grid dimensions (try to make it as square as possible)
    int num_cameras = frames.size();
    int cols = static_cast<int>(std::ceil(std::sqrt(num_cameras)));
    int rows = static_cast<int>(std::ceil(static_cast<double>(num_cameras) / cols));

    // Get the size of individual frames (assume all frames have the same size)
    cv::Size frame_size = frames[0].size();
    int frame_width = frame_size.width;
    int frame_height = frame_size.height;

    // Create the composite frame
    int composite_width = cols * frame_width;
    int composite_height = rows * frame_height;
    cv::Mat composite_frame = cv::Mat::zeros(composite_height, composite_width, frames[0].type());

    // Copy each frame to its position in the composite
    for (int i = 0; i < num_cameras; ++i)
    {
        int row = i / cols;
        int col = i % cols;

        int x = col * frame_width;
        int y = row * frame_height;

        cv::Rect roi(x, y, frame_width, frame_height);

        if (!frames[i].empty() && frames[i].size() == frame_size)
        {
            frames[i].copyTo(composite_frame(roi));

            // Add camera index label
            std::string label = "Camera " + std::to_string(camera_indices[i]);
            cv::putText(composite_frame,
                        label,
                        cv::Point(x + 10, y + 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1,
                        cv::Scalar(0, 255, 0),
                        2);
        }
        else
        {
            // Fill with black if frame is empty or wrong size
            composite_frame(roi) = cv::Scalar(0, 0, 0);

            // Add "No Signal" text
            std::string label = "Camera " + std::to_string(camera_indices[i]) + " - No Signal";
            cv::putText(composite_frame,
                        label,
                        cv::Point(x + 10, y + frame_height / 2),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 0, 255),
                        2);
        }
    }

    // Add FPS display in the bottom-left corner
    if (fps > 0.0)
    {
        std::ostringstream fps_text;
        fps_text << "FPS: " << std::fixed << std::setprecision(1) << fps;

        // Calculate text size to position it properly
        int font_face = cv::FONT_HERSHEY_SIMPLEX;
        double font_scale = 1.0;
        int thickness = 2;
        cv::Size text_size =
            cv::getTextSize(fps_text.str(), font_face, font_scale, thickness, nullptr);

        // Position in bottom-left corner with some padding
        cv::Point text_pos(20, composite_height - 20);

        // Add a semi-transparent background for better visibility
        cv::Rect bg_rect(text_pos.x - 10,
                         text_pos.y - text_size.height - 5,
                         text_size.width + 20,
                         text_size.height + 15);
        cv::rectangle(composite_frame, bg_rect, cv::Scalar(0, 0, 0), -1);
        cv::rectangle(composite_frame, bg_rect, cv::Scalar(255, 255, 255), 1);

        // Draw the FPS text
        cv::putText(composite_frame,
                    fps_text.str(),
                    text_pos,
                    font_face,
                    font_scale,
                    cv::Scalar(0, 255, 255),
                    thickness);
    }

    return composite_frame;
}

int main(int argc, char** argv)
{
    AppOptions options;
    CameraConfig camera_config;

    CLI::App app("libcamera_opencv example");

    // Disable default -h option and use --help only
    app.set_help_flag("--help", "Show help message");

    app.add_option("-c,--camera",
                   options.camera_indices,
                   "Specify which cameras to operate on by index (e.g., -c 0 -c 1 or -c 0,1)")
        ->default_val(std::vector<int> {1});

    app.add_option("-t,--timeout",
                   options.timeout,
                   "duration to capture frame(seconds), 0 - infinit")
        ->default_val(0);

    app.add_flag(
        "--no-window",
        [&options](int count) {
            options.show_window = !count;
        },
        "Disable display window");

    app.add_option("-w,--width", camera_config.width, "Set frame width");
    app.add_option("-h,--height", camera_config.height, "Set frame height");
    app.add_option("-f,--fps", camera_config.frame_rate, "Set frame rate");
    app.add_flag("-v,--verbose", options.verbose, "Enable verbose logging")->default_val(true);
    app.add_flag("--hflip", camera_config.hflip, "Enable horizontal flip");
    app.add_flag("--vflip", camera_config.vflip, "Enable vertical flip");

    CLI11_PARSE(app, argc, argv);

    cv_logging::setLogLevel(
        options.verbose ? cv_logging::LOG_LEVEL_INFO : cv_logging::LOG_LEVEL_WARNING);

    // Register signal handler for Ctrl-C
    std::signal(SIGINT, handle_sigint);

    // Create multiple camera capture objects
    const int camera_count = options.camera_indices.size();
    cv::CameraCapture cameras[camera_count];

    for (int i = 0; i < camera_count; ++i)
    {
        const int camera_index = options.camera_indices[i];
        if (!configureCameraCapture(cameras[i], camera_index, camera_config))
        {
            CV_LOG_ERROR(NULL, "Failed to configure camera " << camera_index);
            return -1;
        }
    }

    if (options.show_window)
    {
        cv::namedWindow("LibCamera Multi-View", cv::WINDOW_AUTOSIZE);
        cv::setWindowTitle("LibCamera Multi-View", "Press q to exit");
    }

    // Record start time for duration control
    const auto start_time = std::chrono::steady_clock::now();
    int frame_count = 0;

    // Real-time FPS calculation variables
    auto last_fps_update = start_time;
    int frames_since_last_update = 0;
    double current_fps = 0.0;

    std::vector<cv::Mat> frames(camera_count);
    while (!stop_flag)
    {
        // Check if capture duration has elapsed
        if (options.timeout > 0)
        {
            const auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= std::chrono::seconds(options.timeout))
            {
                CV_LOG_INFO(NULL, "Capture timeout");
                break;
            }
        }

        bool any_failure = false;
        // issue capture request at same time
        for (size_t i = 0; i < camera_count; ++i)
        {
            if (!cameras[i].grab())
            {
                any_failure = true;
                CV_LOG_ERROR(NULL, "Failed to grab from camera " << options.camera_indices[i]);
                break;
            }
        }

        if (any_failure)
            break;

        for (size_t i = 0; i < camera_count; ++i)
        {
            if (!cameras[i].retrieve(frames[i]))
            {
                any_failure = true;
                CV_LOG_WARNING(
                    NULL,
                    "Failed to retrieve frame from camera " << options.camera_indices[i]);
                break;
            }
        }

        if (any_failure)
            break;

        // Display the merged frame
        if (options.show_window)
        {
            cv::Mat merged_frame = mergeFrames(frames, options.camera_indices, current_fps);
            cv::imshow("LibCamera Multi-View", merged_frame);
            // Check for 'q' key press to exit early
            if (cv::pollKey() == 'q')
            {
                CV_LOG_INFO(NULL, "Capture stopped by user");
                break;
            }
        }

        frame_count++;
        frames_since_last_update++;

        // Update real-time FPS every 2 seconds
        auto current_time = std::chrono::steady_clock::now();
        auto fps_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_update);
        if (fps_elapsed.count() >= 2000)  // 2 seconds
        {
            current_fps = frames_since_last_update * 1000.0 / fps_elapsed.count();
            CV_LOG_INFO(NULL,
                        "Real-time FPS: " << std::fixed << std::setprecision(1) << current_fps);

            // Reset counters
            last_fps_update = current_time;
            frames_since_last_update = 0;
        }
    }

    // Calculate and display FPS statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = total_elapsed.count() / 1000.0;
    double actual_fps = frame_count * 1000 / total_elapsed.count();

    CV_LOG_INFO(NULL, "=== Capture Statistics ===");
    CV_LOG_INFO(NULL, "Total frames captured: " << frame_count);
    CV_LOG_INFO(NULL, "Total elapsed time: " << elapsed_seconds << " seconds");
    CV_LOG_INFO(NULL, "Average FPS: " << std::fixed << std::setprecision(2) << actual_fps);
    if (current_fps > 0)
    {
        CV_LOG_INFO(
            NULL,
            "Last measured real-time FPS: " << std::fixed << std::setprecision(1) << current_fps);
    }

    // Clean up
    for (auto& camera : cameras)
    {
        camera.release();
    }

    if (options.show_window)
        cv::destroyAllWindows();

    CV_LOG_INFO(NULL, "Capture completed");
    return 0;
}
