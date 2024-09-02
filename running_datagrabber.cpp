// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2017 Intel Corporation. All Rights Reserved.
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <librealsense2/rs.hpp> // Include RealSense Cross Platform API
#include <iostream>
#include <map>
#include <chrono>
#include <mutex>
#include <thread>
#include <eigen3/Eigen/Core>
#include <mutex>
#include <unistd.h>
#include <filesystem>
#include <ctime>
#include <sstream>
#include <fstream>
enum IMU{
    ACC,
    GYR
};
struct data_t{
    IMU type;
    Eigen::Vector3f data;
    data_t() = default;
    data_t(const IMU t, const Eigen::Vector3f& d):type(t), data(d){}
};
// The callback example demonstrates asynchronous usage of the pipeline
int main(int argc, char * argv[]) try
{
    //rs2::log_to_console(RS2_LOG_SEVERITY_ERROR);
    std::map<int, int> counters;
    std::map<int, std::string> stream_names;
    std::mutex mutex;

    // Define frame callback
    // The callback is executed on a sensor thread and can be called simultaneously from multiple sensors
    // Therefore any modification to common memory should be done under lock
        // Add streams of gyro and accelerometer to configuration
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);

    std::mutex mtx_mat, mtx_imu;
    std::map<unsigned long long, cv::Mat> m_timestampns_Mat;
    std::map<unsigned long long, data_t> m_timestampns_Imu;
    
    auto callback = [&](const rs2::frame& frame)
    {
        if (rs2::frameset fs = frame.as<rs2::frameset>())
        {
            // With callbacks, all synchronized stream will arrive in a single frameset
            for (const rs2::frame& f : fs){
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    counters[f.get_profile().unique_id()]++;
                }
                // std::cout<<"color "<<f.get_profile().unique_id()<<std::endl;
                // just color 
                
                double timestamp_ms = f.get_timestamp();
                cv::Mat cvf(cv::Size(640, 480), CV_8UC3, (void*)f.get_data());
                unsigned long long t_ns = (unsigned long long)(timestamp_ms * 1e6);
                {
                    std::lock_guard<std::mutex> lock(mtx_mat);
                    m_timestampns_Mat[t_ns] = cvf;
                }
            }
        }
        else
        {
            // Stream that bypass synchronization (such as IMU) will produce single frames
            {
                std::lock_guard<std::mutex> lock(mutex);
                counters[frame.get_profile().unique_id()]++;
            }
            // std::cout<<"imu "<<frame.get_profile().unique_id()<<std::endl;
            // has imu acc and gyr
            auto motion = frame.as<rs2::motion_frame>();
            if (motion && motion.get_profile().stream_type() == RS2_STREAM_GYRO && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
            {
                // Get the timestamp of the current frame
                double tms = motion.get_timestamp();
                unsigned long long t_ns = (unsigned long long)(tms * 1e6);
                // Get gyro measures
                rs2_vector gyro_data = motion.get_motion_data();
                // Call function that computes the angle of motion based on the retrieved measures
                Eigen::Vector3f gyr(gyro_data.x, gyro_data.y, gyro_data.z);
                {
                    std::lock_guard<std::mutex> lock(mtx_imu);
                    m_timestampns_Imu[t_ns] = data_t(IMU::GYR, gyr);
                }
            }
            // If casting succeeded and the arrived frame is from accelerometer stream
            if (motion && motion.get_profile().stream_type() == RS2_STREAM_ACCEL && motion.get_profile().format() == RS2_FORMAT_MOTION_XYZ32F)
            {
                double tms = motion.get_timestamp();
                unsigned long long t_ns = (unsigned long long)(tms * 1e6);
                // Get accelerometer measures
                rs2_vector accel_data = motion.get_motion_data();
                // Call function that computes the angle of motion based on the retrieved measures
                Eigen::Vector3f acc(accel_data.x, accel_data.y, accel_data.z);
                {
                    std::lock_guard<std::mutex> lock(mtx_imu);
                    m_timestampns_Imu[t_ns] = data_t(IMU::ACC, acc);
                }
            }
        }
    };

    // Declare RealSense pipeline, encapsulating the actual device and sensors.
    rs2::pipeline pipe;

    // Start streaming through the callback with default recommended configuration
    // The default video configuration contains Depth and Color streams
    // If a device is capable to stream IMU data, both Gyro and Accelerometer are enabled by default
    //
    rs2::pipeline_profile profiles = pipe.start(cfg, callback);

    // Collect the enabled streams names
    for (auto p : profiles.get_streams())
        stream_names[p.unique_id()] = p.stream_name();

    std::cout << "RealSense callback sample" << std::endl << std::endl;

    while (char(cv::waitKey(1)) != 'q')
    {
        {
        std::lock_guard<std::mutex> lock(mutex);
        std::cout << "\r";
        for (auto p : counters)
        {
            std::cout << stream_names[p.first] << "[" << p.first << "]: " << p.second << " [frames] || ";
        }   
        }
        {
            std::lock_guard<std::mutex> lock(mtx_mat);
            std::cout<<"\nframes "<<m_timestampns_Mat.size();
            if(!m_timestampns_Mat.empty()){
                auto iter = m_timestampns_Mat.end();
                iter--;
                cv::Mat img = iter->second;
                cv::imshow("img", img);
            }
        }
        {
            std::lock_guard<std::mutex> lock(mtx_imu);
            std::cout<<"\nImu "<<m_timestampns_Imu.size();
        }
    }
    pipe.stop();
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d_%H-%M-%S");
    std::string timeStr = ss.str();
    std::filesystem::create_directory(timeStr);
    std::filesystem::create_directory(timeStr+"/color");
    const std::string fpMat = "./"+timeStr+"/color/";
    // create filepath 
    for(auto tMat : m_timestampns_Mat){
        std::string fn = std::to_string(tMat.first);
        cv::imwrite(fpMat+fn+".png", tMat.second);
    }

    const std::string fpImu = "./"+timeStr+"/imu.txt";
    std::ofstream writer(fpImu);
    if(writer.is_open()){
        for (auto tImu : m_timestampns_Imu)
        {
            writer << std::to_string(tImu.first)<<" "<<tImu.second.type<<" "<<tImu.second.data.transpose()<<std::endl;
        }
        writer.close();
    }
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}
