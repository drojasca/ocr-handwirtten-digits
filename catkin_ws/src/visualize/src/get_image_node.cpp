#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>

int main (int argc, char **argv) {
    ros::init(argc, argv, "getImagePath");
    ros::NodeHandle n;

    ros::Publisher image = n.advertise<std_msgs::String>("getImage", 1000);
    ros::Rate loop_rate(10);

    while (ros::ok()) {
        std_msgs::String msg;
        std::string path;
        std::cout << "Enter image path: ";
        std::cin >> path;
        msg.data = path;
        image.publish(msg);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;

}