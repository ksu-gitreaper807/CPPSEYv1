#pragma once

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

// One helper process owns the portal session and sends timestamped BGRx frames.
// An ordinary pipe exposes EOF when it exits; no FIFO anchors or shell commands.
struct WaylandCapture {
    pid_t child_pid = -1;
    int data_fd = -1, log_fd = -1;
    int width = 0, height = 0;
    int64_t last_pts = -1;
    bool active = false, failed = false;
    std::string messages;
    WaylandCapture() = default;
    WaylandCapture(const WaylandCapture&) = delete;
    WaylandCapture& operator=(const WaylandCapture&) = delete;
    ~WaylandCapture() { cleanup(); }

    void drain_logs() {
        if (log_fd < 0) return;
        char bytes[4096];
        ssize_t n;
        while ((n = read(log_fd, bytes, sizeof(bytes))) > 0)
            messages.append(bytes, static_cast<size_t>(n));
        size_t end;
        while ((end = messages.find('\n')) != std::string::npos) {
            const std::string line = messages.substr(0, end);
            messages.erase(0, end + 1);
            if (line.rfind("ERROR ", 0) == 0) failed = true;
            if (line.rfind("READY ", 0) == 0) {
                char extra;
                if (sscanf(line.c_str(), "READY %dx%d%c", &width, &height, &extra) != 2
                    || width < 2 || height < 2 || (width & 1) || (height & 1)
                    || width > 16384 || height > 16384) {
                    std::cerr << "[!] Invalid portal frame geometry.\n";
                    failed = true;
                } else active = true;
            } else if (!line.empty()) std::cerr << "[portal] " << line << "\n";
        }
        if (messages.size() > 65536) {
            std::cerr << "[!] Portal diagnostic line exceeded limit.\n";
            failed = true;
            messages.clear();
        }
    }

    bool read_exact(void* output, size_t count) {
        auto* dest = static_cast<uint8_t*>(output);
        auto progress = std::chrono::steady_clock::now();
        while (count && !g_shutdown_requested.load(std::memory_order_relaxed)) {
            drain_logs();
            if (failed) return false;
            pollfd pfd{data_fd, POLLIN, 0};
            const int ready = poll(&pfd, 1, 100);
            if (ready < 0) {
                if (errno == EINTR) continue;
                failed = true;
                return false;
            }
            if (ready > 0) {
                const ssize_t n = read(data_fd, dest, count);
                if (n > 0) {
                    dest += n;
                    count -= static_cast<size_t>(n);
                    progress = std::chrono::steady_clock::now();
                    continue;
                }
                if (n == 0) {
                    drain_logs();
                    std::cerr << "[!] Portal stream closed.\n";
                    failed = true;
                    return false;
                }
                if (errno != EAGAIN && errno != EINTR) {
                    failed = true;
                    return false;
                }
            }
            if (std::chrono::steady_clock::now() - progress > std::chrono::seconds(15)) {
                std::cerr << "[!] No portal frame data for 15 seconds.\n";
                failed = true;
                return false;
            }
        }
        return count == 0;
    }

    int read_packet(AVPacket* packet) {
        uint8_t header[16];
        if (!read_exact(header, sizeof(header))) return AVERROR_EOF;
        const auto little_endian = [](const uint8_t* p, int n) {
            uint64_t value = 0;
            for (int i = 0; i < n; ++i) value |= uint64_t(p[i]) << (8 * i);
            return value;
        };
        const uint64_t size = little_endian(header + 4, 4);
        const uint64_t pts = little_endian(header + 8, 8);
        const uint64_t expected = uint64_t(width) * height * 4;
        if (memcmp(header, "PSE1", 4) != 0 || size != expected
            || pts > uint64_t(INT64_MAX) || (last_pts >= 0 && pts <= uint64_t(last_pts))) {
            std::cerr << "[!] Invalid portal frame header, size, or timestamp.\n";
            failed = true;
            return AVERROR_INVALIDDATA;
        }
        if (av_new_packet(packet, static_cast<int>(size)) < 0) {
            failed = true;
            return AVERROR(ENOMEM);
        }
        if (!read_exact(packet->data, size)) {
            av_packet_unref(packet);
            return AVERROR_EOF;
        }
        last_pts = static_cast<int64_t>(pts);
        packet->pts = packet->dts = last_pts;
        packet->stream_index = 0;
        packet->flags |= AV_PKT_FLAG_KEY;
        return 0;
    }

    void cleanup() {
        if (data_fd >= 0) { close(data_fd); data_fd = -1; }
        if (child_pid > 0) {
            kill(child_pid, SIGTERM);
            int status = 0;
            bool reaped = false;
            for (int i = 0; i < 50; ++i) {
                const pid_t done = waitpid(child_pid, &status, WNOHANG);
                if (done == child_pid || (done < 0 && errno == ECHILD)) {
                    reaped = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if (!reaped) {
                kill(child_pid, SIGKILL);
                while (waitpid(child_pid, &status, 0) < 0 && errno == EINTR) {}
            }
            child_pid = -1;
        }
        drain_logs();
        if (log_fd >= 0) { close(log_fd); log_fd = -1; }
        active = false;
    }
};

static void wayland_capture_cleanup(WaylandCapture* wc) {
    if (wc) wc->cleanup();
}

static std::string pse_script_path() {
    char path[4096] = {};
    const ssize_t n = readlink("/proc/self/exe", path, sizeof(path) - 1);
    const std::string executable = n > 0 ? std::string(path, n) : "./detector";
    return executable.substr(0, executable.rfind('/') + 1) + "pse_portal_capture.py";
}

static AVFormatContext* open_wayland_capture(const char* framerate,
                                             std::string& video_size_storage,
                                             WaylandCapture& wc) {
    const std::string script = pse_script_path();
    if (access(script.c_str(), R_OK) != 0) {
        std::cerr << "[!] Missing portal helper: " << script << "\n";
        return nullptr;
    }
    int frames[2], logs[2];
    if (pipe2(frames, O_CLOEXEC) < 0) return nullptr;
    if (pipe2(logs, O_CLOEXEC) < 0) {
        close(frames[0]); close(frames[1]);
        return nullptr;
    }
    const pid_t pid = fork();
    if (pid == 0) {
        if (dup2(frames[1], STDOUT_FILENO) < 0 || dup2(logs[1], STDERR_FILENO) < 0)
            _exit(126);
        close(frames[0]); close(frames[1]);
        close(logs[0]); close(logs[1]);
        const char* test_frames = std::getenv("PSE_MAX_FRAMES");
        if (std::getenv("PSE_TEST_CAPTURE")
            && std::strcmp(std::getenv("PSE_TEST_CAPTURE"), "1") == 0) {
            if (test_frames && *test_frames)
                execlp("python3", "python3", "-u", script.c_str(), "--test-source",
                       "--fps", framerate, "--frames", test_frames, nullptr);
            execlp("python3", "python3", "-u", script.c_str(), "--test-source",
                   "--fps", framerate, nullptr);
        }
        execlp("python3", "python3", "-u", script.c_str(), "--fps", framerate, nullptr);
        _exit(127);
    }
    close(frames[1]); close(logs[1]);
    if (pid < 0) {
        close(frames[0]); close(logs[0]);
        return nullptr;
    }
    wc.child_pid = pid;
    wc.data_fd = frames[0];
    wc.log_fd = logs[0];
    fcntl(wc.data_fd, F_SETFL, fcntl(wc.data_fd, F_GETFL) | O_NONBLOCK);
    fcntl(wc.log_fd, F_SETFL, fcntl(wc.log_fd, F_GETFL) | O_NONBLOCK);
    const bool test_source = std::getenv("PSE_TEST_CAPTURE")
                          && std::strcmp(std::getenv("PSE_TEST_CAPTURE"), "1") == 0;
    std::cerr << (test_source
        ? "[*] Using GStreamer test video; no screen portal will open.\n"
        : "[*] Wayland ScreenCast portal: select a physical monitor in the sharing dialog.\n");
    const auto started = std::chrono::steady_clock::now();
    while (!wc.active && !wc.failed && !g_shutdown_requested.load()) {
        wc.drain_logs();
        if (wc.active || wc.failed) break;
        pollfd pfd{wc.log_fd, POLLIN, 0};
        const int ready = poll(&pfd, 1, 100);
        if (ready < 0 && errno != EINTR) { wc.failed = true; break; }
        if (ready > 0 && (pfd.revents & POLLHUP)) {
            wc.drain_logs();
            if (!wc.active) {
                std::cerr << "[!] Portal helper exited before its first frame.\n";
                wc.failed = true;
            }
        }
        if (std::chrono::steady_clock::now() - started > std::chrono::seconds(210)) {
            std::cerr << "[!] Screen sharing was not started within 210 seconds.\n";
            wc.failed = true;
        }
    }
    if (!wc.active || wc.failed) { wc.cleanup(); return nullptr; }
    video_size_storage = std::to_string(wc.width) + "x" + std::to_string(wc.height);
    AVFormatContext* ctx = avformat_alloc_context();
    AVStream* stream = ctx ? avformat_new_stream(ctx, nullptr) : nullptr;
    if (!stream) { avformat_free_context(ctx); wc.cleanup(); return nullptr; }
    stream->time_base = AVRational{1, 1000000000};
    stream->avg_frame_rate = stream->r_frame_rate = av_d2q(std::strtod(framerate, nullptr), 1000000);
    auto* par = stream->codecpar;
    par->codec_type = AVMEDIA_TYPE_VIDEO;
    par->codec_id = AV_CODEC_ID_RAWVIDEO;
    par->format = AV_PIX_FMT_BGR0;
    par->width = wc.width;
    par->height = wc.height;
    std::cerr << "[*] Portal capture live: " << video_size_storage
              << ", source timestamps preserved, up to " << framerate << " fps.\n";
    return ctx;
}
