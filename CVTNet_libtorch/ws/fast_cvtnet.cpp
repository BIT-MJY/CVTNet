#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include <vector>
#include <chrono>
#include <string>
#include <vector>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <pcl/point_cloud.h>

#include <stdio.h>
#include <stdlib.h>
#include<fstream>
#include<string>
#include <sys/types.h>
#include <dirent.h>
using namespace std;


std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file(lidar_data_path, std::ifstream::in | std::ifstream::binary);
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

void gen_range_image(float * virtual_image, pcl::PointCloud<pcl::PointXYZI>::Ptr current_vertex,
                        float fov_up, float fov_down, int proj_H, int proj_W, int max_range, float farer_bound, float nearer_bound)
{
    int len_arr = proj_W*proj_H;
    float fov = std::abs(fov_down) + std::abs(fov_up);

    for(int i=0;i<len_arr;i++)
	    virtual_image[i] = -1.0;

    for(int p=0; p<current_vertex->points.size(); p++)
    {
        float px = current_vertex->points[p].x;
        float py = current_vertex->points[p].y;
        float pz = current_vertex->points[p].z;
        float depth = sqrt(px*px+py*py+pz*pz);

        if (depth > farer_bound || depth < nearer_bound) continue;

        float yaw = -std::atan2(py, px);
        float pitch = std::asin(pz / depth);

        float proj_x = 0.5 * (yaw / M_PI + 1.0);
        float proj_y = 1.0 - (pitch + std::abs(fov_down)) / fov;

        proj_x *= proj_W;
        proj_y *= proj_H;

        proj_x = std::floor(proj_x);
        proj_x = std::min(proj_W - 1, static_cast<int>(proj_x));
        proj_x = std::max(0, static_cast<int>(proj_x));

        proj_y = std::floor(proj_y);
        proj_y = std::min(proj_H - 1, static_cast<int>(proj_y));
        proj_y = std::max(0, static_cast<int>(proj_y));

        float old_depth = virtual_image[int(proj_y*proj_W + proj_x)];
        if ((depth < old_depth && old_depth > 0) || old_depth < 0)
        {
            virtual_image[int(proj_y*proj_W + proj_x)] = depth;
        }

    }
}


void gen_bev_image(float * virtual_image, pcl::PointCloud<pcl::PointXYZI>::Ptr current_vertex,
                        float fov_up, float fov_down, int proj_H, int proj_W, int max_range, float upper_bound, float lower_bound)
{
    int len_arr = proj_W*proj_H;
    float fov = std::abs(fov_down) + std::abs(fov_up);

    for(int i=0;i<len_arr;i++)
	    virtual_image[i] = -1.0;

    for(int p=0; p<current_vertex->points.size(); p++)
    {
        float px = current_vertex->points[p].x;
        float py = current_vertex->points[p].y;
        float pz = current_vertex->points[p].z;
        float depth = sqrt(px*px+py*py+pz*pz);

        if (depth > max_range) continue;
        if (pz > upper_bound || pz < lower_bound) continue;

        float yaw = -std::atan2(py, px);
        float pitch = std::asin(pz / depth);

        float scan_r = depth * std::cos(pitch);
        float proj_x = 0.5 * (yaw / M_PI + 1.0);
        float proj_y = scan_r / max_range;

        proj_x *= proj_W;
        proj_y *= proj_H;

        proj_x = std::floor(proj_x);
        proj_x = std::min(proj_W - 1, static_cast<int>(proj_x));
        proj_x = std::max(0, static_cast<int>(proj_x));

        proj_y = std::floor(proj_y);
        proj_y = std::min(proj_H - 1, static_cast<int>(proj_y));
        proj_y = std::max(0, static_cast<int>(proj_y));

        float old_height = virtual_image[int(proj_y*proj_W + proj_x)];
        if ((pz > old_height && old_height > 0) || old_height < 0)
        {
            virtual_image[int(proj_y*proj_W + proj_x)] = pz;
        }

    }
}



int main()
{

    torch::DeviceType device_type;
    device_type = torch::kCUDA;
    torch::Device device(device_type);
    std::cout<<"cuda support:"<< (torch::cuda::is_available()?"ture":"false")<<std::endl;
    torch::jit::script::Module module = torch::jit::load("../../CVTNet.pt");
    module.to(torch::kCUDA);
    module.eval();

    std::string lidar_file = "../1.pcd";
    std::vector<float> lidar_data = read_lidar_data(lidar_file);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZI>);

    for (std::size_t i = 0; i < lidar_data.size(); i += 1)
    {
        pcl::PointXYZI point;
        point.x = lidar_data[i];
        point.y = lidar_data[i + 1];
        point.z = lidar_data[i + 2];
        point.intensity = lidar_data[i + 3];
        cloud0->points.push_back(point);
    }

    int width = 900;
    int height = 32;
    std::vector<float> range_thresh = {0.0, 15.0, 30.0, 45.0, 60.0};
    std::vector<float> height_thresh = {-4.0, 0.0, 4.0, 8.0, 12.0};
    int interval = range_thresh.size();
    int len_arr = width*height;
    int len_arr_all = 2*interval*width*height;
    float ri_bev_image[len_arr_all];

    for(int th=0; th<range_thresh.size(); th++)
    {
        float range_image[len_arr];
        float farer_bound = range_thresh[th+1];
        float nearer_bound = range_thresh[th];
        if (th == range_thresh.size()-1)
        {
            gen_range_image(range_image, cloud0, 15, -16, 32, 900, 80, range_thresh[int(interval-1)], range_thresh[0]);
        }
        else
        {
            gen_range_image(range_image, cloud0, 15, -16, 32, 900, 80, farer_bound, nearer_bound);
        }
        for(int x=0; x<width; x++)
        {
            for(int y=0; y<height; y++)
            {
                int idx_in_all = int(th*width*height + y*width + x);
                ri_bev_image[idx_in_all] = range_image[y*width + x];
            }
        }
    }

    for(int th=0; th<height_thresh.size(); th++)
    {
        float bev_image[len_arr];
        float upper_bound = height_thresh[th+1];
        float lower_bound = height_thresh[th];
        if (th == height_thresh.size()-1)
        {
            gen_bev_image(bev_image, cloud0, 15, -16, 32, 900, 80, height_thresh[int(interval-1)], height_thresh[0]);
        }
        else
        {
            gen_bev_image(bev_image, cloud0, 15, -16, 32, 900, 80, upper_bound, lower_bound);
        }

        for(int x=0; x<width; x++)
        {
            for(int y=0; y<height; y++)
            {
                int idx_in_all = int((th+5)*width*height + y*width + x);
                ri_bev_image[idx_in_all] = bev_image[y*width + x];
            }
        }
    }

    torch::Tensor tester  = torch::from_blob(ri_bev_image, {1, 10, 32, 900}, torch::kFloat).to(device);
    torch::Tensor result = module.forward({tester}).toTensor();
    result = result.to(torch::kCPU);
    std::cout<<result<<std::endl;

    return 0;
}