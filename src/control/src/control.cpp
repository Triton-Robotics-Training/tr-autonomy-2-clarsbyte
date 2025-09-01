#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using std::placeholders::_1;
using namespace std::chrono_literals;

class Control : public rclcpp::Node
{
public:
    Control() : Node("control"), count_(0), x(0), y(0), current_angle_(0.0), last_good_angle_(0.0), 
                last_target_time_(this->now()), centered_start_time_(this->now()), is_centered_(false), 
                points_scored_(0)
    {
        publisher_ = this->create_publisher<std_msgs::msg::Float32>("desired_angle", 10);
        image_publisher_ = this->create_publisher<sensor_msgs::msg::Image>("processed_image", 10);
        
        image_subscriber_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/robotcam", 10,
            std::bind(&Control::imageCallback, this, std::placeholders::_1));
            
        angle_subscriber_ = this->create_subscription<std_msgs::msg::Float32>(
            "/current_angle", 10,
            std::bind(&Control::angleCallback, this, std::placeholders::_1));
        
        timer_ = this->create_wall_timer(std::chrono::milliseconds(100), 
            std::bind(&Control::controlLoop, this));
        
        cv::namedWindow(OPENCV_WINDOW);
            }

    ~Control()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }

private:
    static constexpr const char* OPENCV_WINDOW = "Robot Vision";
    std::vector<uint8_t> pixels; 
    int x, y;  
    float current_angle_;
    float last_good_angle_;  
    rclcpp::Time last_target_time_;  
    rclcpp::Time centered_start_time_;  
    bool is_centered_; 
    int points_scored_; 
    cv::Mat received_image_;  
    rclcpp::TimerBase::SharedPtr timer_; 
    float target_center_x_;  
    
    void angleCallback(const std_msgs::msg::Float32::ConstSharedPtr msg)
    {
        current_angle_ = msg->data;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
            "Current robot angle: %.3f rad", current_angle_);
    }  
    
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
    {
        try
        {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            received_image_ = cv_ptr->image.clone();
        }
        catch (cv_bridge::Exception& e)
        {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }
    
    void controlLoop()
    {
        if (received_image_.empty()) return;
        
        // Convert to HSV 
        cv::Mat hsv_image;
        cv::cvtColor(received_image_, hsv_image, cv::COLOR_BGR2HSV);
        
        cv::Mat mask;
        cv::Scalar lower_bound = cv::Scalar(0, 0, 0);
        cv::Scalar upper_bound = cv::Scalar(179, 109, 255);
        cv::inRange(hsv_image, lower_bound, upper_bound, mask);
        
        // Invert mask 
        cv::Mat inverted_mask;
        cv::bitwise_not(mask, inverted_mask);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::Mat hierarchy;
        cv::findContours(inverted_mask, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        float desired_angle = 0.0f;
        
        if (!contours.empty()) {
            std::vector<cv::Point> target_contour = contours[0];
            
            cv::Moments moment = cv::moments(target_contour);
            target_center_x_ = moment.m10 / moment.m00;
            float target_center_y = moment.m01 / moment.m00;
            
            float half_image_width = received_image_.cols / 2.0f;
            float pixel_error = target_center_x_ - half_image_width;
            
            float ratio_of_target = pixel_error / half_image_width;
            float predicted_angle_movement = -1.0f * ratio_of_target * (M_PI/2.0f); 
            
            desired_angle = current_angle_ + predicted_angle_movement;
            
            if (desired_angle > M_PI/2.0) desired_angle = M_PI/2.0;
            if (desired_angle < -M_PI/2.0) desired_angle = -M_PI/2.0;
            
            bool target_centered = (abs(pixel_error) <= 60);
            
            if (target_centered) {
                if (!is_centered_) {
                    is_centered_ = true;
                    centered_start_time_ = this->now();
                } else {
                    auto centered_duration = this->now() - centered_start_time_;
                    if (centered_duration.seconds() >= 2.0) {
                        points_scored_++;
                        RCLCPP_INFO(this->get_logger(), "POINT SCORED");
                        is_centered_ = false;
                        desired_angle = 0.0f;  // Home to 0 after scoring
                    }
                }
            } else {
                is_centered_ = false;
            }
            
            cv::circle(received_image_, cv::Point(target_center_x_, target_center_y), 10, cv::Scalar(0, 255, 0), 3);
            
            int image_center = received_image_.cols / 2;
            cv::rectangle(received_image_, 
                         cv::Point(image_center - 60, 0), 
                         cv::Point(image_center + 60, received_image_.rows), 
                         cv::Scalar(255, 255, 0), 2);
            
        } else {
            // No target found - reset to 0 
            is_centered_ = false;
            desired_angle = 0.0f;
        }
        
        int center = received_image_.cols / 2;
        cv::line(received_image_, cv::Point(center, 0), cv::Point(center, received_image_.rows), 
                cv::Scalar(255, 255, 255), 2);
        
        cv::imshow(OPENCV_WINDOW, received_image_);
        cv::waitKey(1);
        
        auto angle_message = std_msgs::msg::Float32();
        angle_message.data = desired_angle;
        publisher_->publish(angle_message);
    }
    
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;
    rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr angle_subscriber_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_publisher_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_;
    size_t count_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Control>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}