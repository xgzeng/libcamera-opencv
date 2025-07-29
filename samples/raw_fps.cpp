#include "camera_capture_impl.hpp"
#include <chrono>
#include <iomanip>
#include <csignal>
#include <atomic>
#include <CLI/CLI.hpp>
#include <opencv2/videoio.hpp>  // CV_CAP_*

using namespace cv;

std::atomic<bool> stop_flag {false};

void handle_sigint(int)
{
    stop_flag = true;
}

int main(int argc, char* argv[])
{
    // Register signal handler for Ctrl-C
    std::signal(SIGINT, handle_sigint);

    std::vector<int> camera_indices = {1};
    std::optional<int> frame_rate;
    std::optional<int> width;
    std::optional<int> height;

    CLI::App app("raw_fps");
    app.set_help_flag("--help", "Show help message");
    app.add_option("-c,--camera", camera_indices, "Specify which cameras to capture")
        ->default_val(std::vector<int> {1});
    app.add_option("-w,--width", width, "Set frame width");
    app.add_option("-h,--height", height, "Set frame height");
    app.add_option("-f,--fps", frame_rate, "Set frame rate");

    CLI11_PARSE(app, argc, argv);

    libcamera::CameraManager camera_manager;
    camera_manager.start();

    std::vector<std::unique_ptr<CameraCaptureImpl>> captures;

    for (auto index : camera_indices)
    {
        if (index <= 0 || index > camera_manager.cameras().size())
        {
            std::cout << "invalid camera index " << index << std::endl;
            continue;
        }
        auto cap = std::make_unique<CameraCaptureImpl>();
        if (!cap->open(camera_manager.cameras()[index - 1]))
            continue;

        if (width)
            cap->setProperty(cv::CAP_PROP_FRAME_WIDTH, width.value());
        if (height)
            cap->setProperty(cv::CAP_PROP_FRAME_HEIGHT, height.value());
        if (frame_rate)
            cap->setProperty(cv::CAP_PROP_FPS, frame_rate.value());
        captures.push_back(std::move(cap));
    }

    // Frame rate calculation variables
    auto start_time = std::chrono::steady_clock::now();
    auto last_fps_update = start_time;
    int frame_count = 0;
    int frames_since_last_update = 0;

    int w = captures[0]->getProperty(cv::CAP_PROP_FRAME_WIDTH);
    int h = captures[0]->getProperty(cv::CAP_PROP_FRAME_HEIGHT);
    std::cout << "Starting capture loop at resolution " << w << "x" << h << std::endl;
    std::cout << "Press Ctrl+C to exit." << std::endl;

    while (!stop_flag)
    {
        for (auto& cap : captures)
        {
            cap->grabFrame(4);
        }

        for (auto& cap : captures)
        {
            libcamera::Request* req = cap->retrieveRequest(100);
            if (!req)
                continue;
            cap->finishRequest(req);
            frame_count++;
            frames_since_last_update++;
        }

        // Update FPS every 2 seconds
        auto current_time = std::chrono::steady_clock::now();
        auto fps_elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_fps_update);
        if (fps_elapsed.count() >= 2000)  // 2 seconds
        {
            double current_fps = frames_since_last_update * 1000.0 / fps_elapsed.count();
            std::cout << "FPS(SUM): " << std::fixed << std::setprecision(1) << current_fps << "; ";
            std::cout << "FPS(CAM): " << std::fixed << std::setprecision(1)
                      << (current_fps / captures.size()) << std::endl;

            // Reset counters
            last_fps_update = current_time;
            frames_since_last_update = 0;
        }
    }

    // Print final statistics
    auto end_time = std::chrono::steady_clock::now();
    auto total_elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    double total_seconds = total_elapsed.count() / 1000.0;
    double final_avg_fps = frame_count / total_seconds;

    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total runtime: " << std::fixed << std::setprecision(2) << total_seconds
              << " seconds" << std::endl;
    std::cout << "Final FPS(SUM): " << std::fixed << std::setprecision(2) << final_avg_fps
              << std::endl;
    std::cout << "Final FPS(CAM): " << std::fixed << std::setprecision(2)
              << (final_avg_fps / captures.size()) << std::endl;

    for (auto& cap : captures)
    {
        cap->close();
    }
    camera_manager.stop();
    return 0;
}