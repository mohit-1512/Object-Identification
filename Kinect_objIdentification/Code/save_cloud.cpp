#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

PointCloudT::Ptr cloud (new PointCloudT);

boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Record"));

unsigned int filesSaved = 0;
bool saveCloud(false);
int j=0;
std::string save = "inputcloud";


void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event,
					  void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
		saveCloud = true;
}

void
grabberCallback(const PointCloudT::ConstPtr& cloud)
{
	if (! viewer->wasStopped())
	{
		viewer->removeAllPointClouds();
		pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
		viewer->addPointCloud<PointT> (cloud, rgb, "input_cloud");

		viewer->spinOnce();
	}
	
	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity(); 
	transMat (1,1) = -1;

	if (saveCloud)
	{
		PointCloudT::Ptr new_cloud (new PointCloudT);
		pcl::transformPointCloud(*cloud,*new_cloud,transMat);

		std::stringstream stream;
		stream << save << j << ".pcd";
		std::string filename = stream.str();
		if (pcl::io::savePCDFile(filename, *new_cloud, false) == 0)
		{
			j++;
			std::cout << "Saved " << filename << "." << endl;
		}
		else PCL_ERROR("Problem saving %s.\n", filename.c_str());

		saveCloud = false;
	}
}

int main (int argc, char** argv)
{

	pcl::console::parse_argument (argc, argv, "-j", j);
	pcl::console::parse_argument (argc, argv, "-save", save);
	
	pcl::Grabber* grab = new pcl::OpenNIGrabber ();
	
	viewer->registerKeyboardCallback(keyboardEventOccurred);
	
	// invert correction
	viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0); 
    viewer->spinOnce();

	if (grab == 0)
		return false;

	boost::function<void (const PointCloudT::ConstPtr&)> f =
		boost::bind(&grabberCallback, _1);
	grab->registerCallback (f);
	grab->start ();

	while (! viewer->wasStopped())
		boost::this_thread::sleep(boost::posix_time::seconds(1));
	
	grab->stop ();

	return 0;

}
