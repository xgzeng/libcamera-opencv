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
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

// Overload to accept string parameter (handles strings shorter than 4 characters)
static constexpr uint32_t CV_FOURCC(const std::string& fourcc_str)
{
    char c[4] = {' ', ' ', ' ', ' '};
    for (int i = 0; i < 4; ++i)
    {
        if (fourcc_str[i] == 0)
            break;
        c[i] = fourcc_str[i];
    }
    return (c[0] & 255) + ((c[1] & 255) << 8) + ((c[2] & 255) << 16) + ((c[3] & 255) << 24);
}

std::atomic<bool> stop_flag {false};

void handle_sigint(int)
{
    stop_flag = true;
}

// Structure to hold camera configuration options
struct CameraConfig
{
    std::optional<int> width;
    std::optional<int> height;
    std::optional<double> frame_rate;
    std::string fourcc;
    bool hflip = false;
    bool vflip = false;
};

struct FrameData
{
    std::vector<cv::Mat> frames;
};

cv::Mat mergeFrames(const std::vector<cv::Mat>& frames);

class MultiCamCapturer
{
private:
    std::queue<FrameData> queue_;
    std::mutex mutex_;
    std::condition_variable condition_;
    size_t queue_size_ = 2;
    std::vector<cv::CameraCapture> cameras_;
    std::thread capture_thread_;
    std::atomic<bool> stop_flag_ {false};

    std::atomic<int> capture_count_;
    std::atomic<int> discard_count_;

public:
    int capture_count() const
    {
        return capture_count_;
    }

    int discard_count() const
    {
        return discard_count_;
    }

    MultiCamCapturer(const std::vector<int>& camera_indices, const CameraConfig& camera_config) :
        cameras_(camera_indices.size())
    {
        for (int i = 0; i < cameras_.size(); ++i)
        {
            const int camera_index = camera_indices[i];
            if (!configureCameraCapture(cameras_[i], camera_index, camera_config))
            {
                CV_LOG_ERROR(NULL, "Failed to configure camera " << camera_index);
                throw std::runtime_error("Failed to configure camera");
            }
        }
    }

    ~MultiCamCapturer()
    {
        for (auto& camera : cameras_)
        {
            camera.release();
        }
    }

    void start()
    {
        // Start capture thread
        capture_thread_ = std::thread(&MultiCamCapturer::captureThread, this);
    }

    void stop()
    {
        stop_flag_ = true;
        capture_thread_.join();
    }

    bool waitFrames(FrameData& frame_data, int timeout_ms)
    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (condition_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
                return !queue_.empty() || stop_flag;
            }))
        {
            if (!queue_.empty())
            {
                frame_data = queue_.front();
                queue_.pop();
                return true;
            }
        }
        return false;
    }

private:
    bool
        configureCameraCapture(cv::CameraCapture& cap, int camera_index, const CameraConfig& config)
    {
        if (!cap.open(camera_index))
        {
            CV_LOG_ERROR(NULL, "Could not open camera " << camera_index);
            return false;
        }

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
        if (!config.fourcc.empty())
        {
            cap.set(cv::CAP_PROP_FOURCC, CV_FOURCC(config.fourcc));
        }

        // Apply flip settings for libcamera-based capture
        cap.setOrientation(config.hflip, config.vflip);

        return true;
    }

    // Capture thread function
    void captureThread()
    {
        const auto start_time = std::chrono::steady_clock::now();
        const int camera_count = cameras_.size();

        while (!stop_flag_)
        {
            FrameData frame_data;
            frame_data.frames.resize(camera_count);

            bool any_failure = false;
            // Issue capture request at same time for all cameras
            for (int i = 0; i < camera_count; ++i)
            {
                if (!cameras_[i].grab())
                {
                    any_failure = true;
                    CV_LOG_ERROR(NULL, "Failed to grab from camera " << cameras_[i].index());
                    break;
                }
            }

            if (any_failure)
                break;

            // Retrieve frames from all cameras
            for (int i = 0; i < camera_count; ++i)
            {
                if (!cameras_[i].retrieve(frame_data.frames[i]))
                {
                    any_failure = true;
                    CV_LOG_WARNING(NULL,
                                   "Failed to retrieve frame from camera " << cameras_[i].index());
                    break;
                }
            }

            if (any_failure)
                break;

            capture_count_ += 1;

            // Push to queue
            std::lock_guard<std::mutex> lock(mutex_);
            if (queue_.size() < queue_size_)
            {
                queue_.push(frame_data);
                condition_.notify_one();
            }
            else
            {
                discard_count_ += 1;
            }
        }
        stop_flag_ = true;
        condition_.notify_one();
        CV_LOG_INFO(NULL, "Capture thread exiting");
    }
};

// Function to merge multiple frames into a single composite frame
cv::Mat mergeFrames(const std::vector<cv::Mat>& frames)
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
        frames[i].copyTo(composite_frame(roi));
    }

    return composite_frame;
}

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

// Structure to hold application options
struct AppOptions
{
    std::vector<int> camera_indices = {0};
    int timeout = 0;  // capture duration(seconds), 0 = infinite
    bool show_window = true;
    bool verbose = true;
    std::string output_file;  // Output video file path
};

enum LableCorner
{
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
};

void labelFrame(cv::Mat& frame, const std::string& text, LableCorner corner)
{
    if (frame.empty() || text.empty())
        return;

    // Text properties
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    double font_scale = 1.0;
    int thickness = 2;
    cv::Scalar text_color(0, 255, 0);  // Green
    cv::Scalar bg_color(0, 0, 0);  // Black background
    cv::Scalar border_color(255, 255, 255);  // White border

    // Calculate text size
    cv::Size text_size = cv::getTextSize(text, font_face, font_scale, thickness, nullptr);

    // Calculate position based on corner
    cv::Point text_pos;
    int padding = 10;

    switch (corner)
    {
        case TOP_LEFT: text_pos = cv::Point(padding, text_size.height + padding); break;
        case TOP_RIGHT:
            text_pos =
                cv::Point(frame.cols - text_size.width - padding, text_size.height + padding);
            break;
        case BOTTOM_LEFT: text_pos = cv::Point(padding, frame.rows - padding); break;
        case BOTTOM_RIGHT:
            text_pos = cv::Point(frame.cols - text_size.width - padding, frame.rows - padding);
            break;
    }

        // Add background rectangle for better visibility
#if 0    
    cv::Rect bg_rect(text_pos.x - padding / 2,
                     text_pos.y - text_size.height - padding / 2,
                     text_size.width + padding,
                     text_size.height + padding);
    cv::rectangle(frame, bg_rect, bg_color, -1);
    cv::rectangle(frame, bg_rect, border_color, 1);
#endif
    // Draw the text
    cv::putText(frame, text, text_pos, font_face, font_scale, text_color, thickness);
}

static bool string_ends_with(const std::string_view& str, const std::string_view& substr)
{
    auto pos = str.rfind(substr);
    return pos != std::string_view::npos && str.length() == (pos + substr.length());
}

class AutoVideoWriter
{
public:
    AutoVideoWriter(const std::string& output_file, int fps = 30) :
        output_file_(output_file), fps_(fps)
    {
        if (!string_ends_with(output_file, ".mp4"))
        {
            CV_LOG_ERROR(NULL, "output file must end with .mp4");
            throw std::runtime_error("invalid filename");
        }
    }

    ~AutoVideoWriter()
    {
        video_writer_.release();
        CV_LOG_INFO(NULL, "Video file saved: " << output_file_);
    }

    void write(const cv::Mat& frame)
    {
        if (!video_writer_.isOpened())
        {
            // Use H.264 codec for better compression
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
            if (video_writer_.open(output_file_, fourcc, fps_, frame.size(), true))
            {
                CV_LOG_INFO(NULL, "Video recording started: " << output_file_);
                CV_LOG_INFO(NULL,  //
                            "Video format: " << frame.size().width << "x" << frame.size().height
                                             << " @ " << fps_);
                return;
            }
            else
            {
                CV_LOG_ERROR(NULL, "Failed to open video writer for: " << output_file_);
                return;
            }
        }
        else
        {
            video_writer_.write(frame);
        }
    }

private:
    std::string output_file_;
    int fps_ = 30;
    cv::VideoWriter video_writer_;
};


int main(int argc, char** argv)
{
    AppOptions options;
    CameraConfig camera_config;

    CLI::App app("libcamera_opencv example");

    // Disable default -h option and use --help only
    app.set_help_flag("--help", "Show help message");

    app.add_option("-c,--camera",
                   options.camera_indices,
                   "Specify which cameras to operate on by index (e.g., -c 0 -c 1 or "
                   "-c 0,1)")
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
    app.add_flag("-v,--verbose", options.verbose, "Enable verbose logging")->default_val(false);
    app.add_flag("--hflip", camera_config.hflip, "Enable horizontal flip");
    app.add_flag("--vflip", camera_config.vflip, "Enable vertical flip");
    app.add_option("-p,--pixel-format,--fourcc", camera_config.fourcc, "Pixel formart fourcc")
        ->default_val(std::string {});
    app.add_option("-o,--output",
                   options.output_file,
                   "Output H.264 video file path (e.g., output.mp4)");

    CLI11_PARSE(app, argc, argv);

    cv_logging::setLogLevel(
        options.verbose ? cv_logging::LOG_LEVEL_INFO : cv_logging::LOG_LEVEL_WARNING);

    // Register signal handler for Ctrl-C
    std::signal(SIGINT, handle_sigint);

    MultiCamCapturer capturer(options.camera_indices, camera_config);

    // Initialize VideoWriter if output file is specified
    double fps = camera_config.frame_rate ? camera_config.frame_rate.value() : 30.0;

    std::unique_ptr<AutoVideoWriter> video_writer;
    if (!options.output_file.empty())
    {
        video_writer = std::make_unique<AutoVideoWriter>(options.output_file, fps);
    }

    if (options.show_window)
    {
        cv::namedWindow("LibCamera Multi-View", cv::WINDOW_AUTOSIZE);
        cv::setWindowTitle("LibCamera Multi-View", "Press q to exit");
    }

    capturer.start();

    // Record start time for duration control and FPS calculation
    const auto start_time = std::chrono::steady_clock::now();

    // Real-time FPS calculation variables
    auto last_fps_update = start_time;
    int total_output_count = 0;
    double current_fps = 0.0;

    // Capture FPS calculation variables
    int last_capture_count = 0;
    int last_output_count = 0;

    // main display/processing loop
    while (!stop_flag)
    {
        FrameData current_frame_data;
        if (!capturer.waitFrames(current_frame_data, 2000))
        {
            CV_LOG_ERROR(NULL, "wait for frames from capturer timeout");
            break;
        }

        cv::Mat merged_frame = mergeFrames(current_frame_data.frames);

        // Display the merged frame
        if (options.show_window)
        {
            cv::imshow("LibCamera Multi-View", merged_frame);
            // Check for 'q' key press to exit early
            if (cv::pollKey() == 'q')
            {
                CV_LOG_INFO(NULL, "Display stopped by user");
                stop_flag = true;
                break;
            }
        }

        if (video_writer)
        {
            video_writer->write(merged_frame);
        }

        total_output_count++;

        // Update real-time FPS every 2 seconds
        auto current_time = std::chrono::steady_clock::now();
        auto fps_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_update);
        if (fps_elapsed.count() >= 2000)  // 2 seconds
        {
            current_fps = (total_output_count - last_output_count) * 1000.0 / fps_elapsed.count();
            CV_LOG_INFO(NULL, "Output FPS: " << std::fixed << std::setprecision(1) << current_fps);

            int current_capture_count = capturer.capture_count();
            int frames_captured = current_capture_count - last_capture_count;

            double current_capture_fps = frames_captured * 1000.0 / fps_elapsed.count();
            CV_LOG_INFO(NULL,
                        "Capture FPS: " << std::fixed << std::setprecision(1) << current_capture_fps
                                        << " (Total: " << current_capture_count
                                        << ", Discarded: " << capturer.discard_count() << ")");

            // Reset counters
            last_fps_update = current_time;
            last_capture_count = current_capture_count;
            last_output_count = total_output_count;
        }
    }

    capturer.stop();

    // Calculate and display final capture statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double elapsed_seconds = total_elapsed.count() / 1000.0;

    CV_LOG_INFO(NULL, "=== Capture Statistics ===");
    CV_LOG_INFO(NULL,
                "Total elapsed time: " << std::fixed << std::setprecision(2) << elapsed_seconds
                                       << " seconds");
    CV_LOG_INFO(NULL, "Total frames captured: " << capturer.capture_count());
    CV_LOG_INFO(NULL, "Total frames discarded: " << capturer.discard_count());
    CV_LOG_INFO(NULL,
                "Average capture FPS: " << std::fixed << std::setprecision(1)
                                        << capturer.capture_count() / elapsed_seconds);

    if (options.show_window)
        cv::destroyAllWindows();

    CV_LOG_INFO(NULL, "Capture completed");
    return 0;
}
