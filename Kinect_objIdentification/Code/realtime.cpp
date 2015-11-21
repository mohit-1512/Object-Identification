#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/time.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/console/parse.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/vfh.h>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr/shared_ptr.hpp>
#include <pcl/common/pca.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <boost/thread.hpp>


typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef std::pair<std::string, std::vector<float> > vfh_model;


struct hypothesis
{
	float distance;
	std::string object_name;
	PointCloudT::Ptr cluster;
	std::string cluster_name;
};


boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Display"));

boost::mutex cloud_mutex;

int j=0;

pcl::Grabber* grab = new pcl::OpenNIGrabber ();

////////////////////
// bounding boxes
////////////////////


Eigen::Vector4f centroid;
	
Eigen::Matrix3f covariance;
Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver;
Eigen::Matrix3f eigDx;


Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
pcl::PointCloud<PointT> cPoints;
PointT min_pt, max_pt;



void
keyboardEventOccurred(const pcl::visualization::KeyboardEvent& event,
					  void* nothing)
{
	if (event.getKeySym() == "space" && event.keyDown())
		boost::this_thread::sleep(boost::posix_time::seconds(5));
   
}


void
drawBoundingBox(PointCloudT::Ptr cloud,
				boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
				int z)
{
	
	//Eigen::Vector4f centroid;
	pcl::compute3DCentroid(*cloud, centroid);
	
	//Eigen::Matrix3f covariance;
	computeCovarianceMatrixNormalized(*cloud, centroid, covariance);
	//Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance,
	//Eigen::ComputeEigenvectors);

	eigen_solver.compute(covariance,Eigen::ComputeEigenvectors);
	
//	eigen_solver = boost::shared_ptr<Eigen::SelfAdjointEigenSolver>
//		(covariance,Eigen::ComputeEigenvectors);

	eigDx = eigen_solver.eigenvectors();
    eigDx.col(2) = eigDx.col(0).cross(eigDx.col(1));


	//Eigen::Matrix4f p2w(Eigen::Matrix4f::Identity());
	p2w.block<3,3>(0,0) = eigDx.transpose();
	p2w.block<3,1>(0,3) = -1.f * (p2w.block<3,3>(0,0) * centroid.head<3>());
    //pcl::PointCloud<PointT> cPoints;
    pcl::transformPointCloud(*cloud, cPoints, p2w);


	//PointT min_pt, max_pt;
    pcl::getMinMax3D(cPoints, min_pt, max_pt);
    const Eigen::Vector3f mean_diag = 0.5f*(max_pt.getVector3fMap() + min_pt.getVector3fMap());

	const Eigen::Quaternionf qfinal(eigDx);
    const Eigen::Vector3f tfinal = eigDx*mean_diag + centroid.head<3>();
	
	//viewer->addPointCloud(cloud);
	viewer->addCube(tfinal, qfinal, max_pt.x - min_pt.x, max_pt.y - min_pt.y, max_pt.z - min_pt.z,boost::lexical_cast<std::string>((z+1)*200));

}


void displayCloud(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
				  PointCloudT::Ptr cloud)
{
	//cloud_mutex.lock();
//viewer->removeAllShapes();
	viewer->removeAllPointClouds();
	viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0); 
	pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
	viewer->addPointCloud<PointT>(cloud,rgb,"input_cloud");
	viewer->spinOnce();
	//cloud_mutex.unlock();
}


void allBoxes(boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer,
			  std::vector<hypothesis> final_hypothesis)
{
	//cloud_mutex.lock();
	viewer->removeAllShapes();
	for(int z=0; z<final_hypothesis.size();++z)
	{
		if(final_hypothesis.at(z).distance < 100)
		{
			std::stringstream ddd;
			ddd << final_hypothesis.at(z).object_name;
			ddd << "\n" << "(";
			ddd << final_hypothesis.at(z).distance;
			ddd << ")";

			viewer->addText3D(ddd.str().c_str(),
							 final_hypothesis.at(z).cluster->points[0],
							 0.02,1,1,1,
							 boost::lexical_cast<std::string>(z));
			drawBoundingBox(final_hypothesis.at(z).cluster,viewer,z);
		}
	}

	final_hypothesis.clear();
	viewer->spinOnce();
	//cloud_mutex.unlock();
}




bool
loadFileList (std::vector<vfh_model> &models, const std::string &filename)
{
  ifstream fs;
  fs.open (filename.c_str ());
  if (!fs.is_open () || fs.fail ())
    return (false);

  std::string line;
  while (!fs.eof ())
  {
    getline (fs, line);
    if (line.empty ())
      continue;
    vfh_model m;
    m.first = line;
    models.push_back (m);
  }
  fs.close ();
  return (true);
}

inline void
nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, 
				const vfh_model &model, 
                int k, 
				flann::Matrix<int> &indices, 
				flann::Matrix<float> &distances)
{

	flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
	memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

	indices = flann::Matrix<int>(new int[k], 1, k);
	distances = flann::Matrix<float>(new float[k], 1, k);
	index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
	delete[] p.ptr ();
}




void
grabberCallback(const PointCloudT::ConstPtr& callback_cloud, 
					PointCloudT::Ptr& cloud,
					bool* new_cloud_available_flag)
{
	cloud_mutex.lock();
	*cloud = *callback_cloud;
	*new_cloud_available_flag = true;
	cloud_mutex.unlock();
}


int main (int argc, char** argv)
{	
	PointCloudT::Ptr cloud (new PointCloudT);
	PointCloudT::Ptr new_cloud (new PointCloudT);
	bool new_cloud_available_flag = false;
	//pcl::Grabber* grab = new pcl::OpenNIGrabber ();

	PointCloudT::Ptr ddd;

	boost::function<void (const PointCloudT::ConstPtr&)> f =
		boost::bind(&grabberCallback, _1, cloud, &new_cloud_available_flag);
	grab->registerCallback (f);
	viewer->registerKeyboardCallback(keyboardEventOccurred);
	grab->start ();
	
	bool first_time = true;

	FILE* objects;
	objects = fopen ("objects.txt","a");

	while(!new_cloud_available_flag)
		boost::this_thread::sleep(boost::posix_time::milliseconds(1));

	new_cloud_available_flag=false;


	////////////////////
	// invert correction
	////////////////////
				
	Eigen::Matrix4f transMat = Eigen::Matrix4f::Identity(); 
	transMat (1,1) = -1;

    ////////////////////
	// pass filter
	////////////////////

	PointCloudT::Ptr passed_cloud;
	pcl::PassThrough<PointT> pass;
	passed_cloud = boost::shared_ptr<PointCloudT>(new PointCloudT);

	
	////////////////////
	// voxel grid
	////////////////////
	PointCloudT::Ptr voxelized_cloud;
	voxelized_cloud = boost::shared_ptr<PointCloudT>(new PointCloudT);
	pcl::VoxelGrid<PointT> vg;
	vg.setLeafSize (0.001, 0.001, 0.001);
	

	////////////////////
	// sac segmentation
	////////////////////
	
	PointCloudT::Ptr cloud_f;
	PointCloudT::Ptr cloud_plane;
	PointCloudT::Ptr cloud_filtered;
	cloud_f = boost::shared_ptr<PointCloudT>(new PointCloudT);	
	cloud_plane = boost::shared_ptr<PointCloudT> (new PointCloudT);	
	cloud_filtered = boost::shared_ptr<PointCloudT> (new PointCloudT);

	pcl::SACSegmentation<PointT> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.02);

	////////////////////
	// euclidean clustering
	////////////////////
	std::vector<pcl::PointIndices> cluster_indices;
	std::vector<PointCloudT::Ptr> extracted_clusters;
	pcl::search::KdTree<PointT>::Ptr eucl_tree (new pcl::search::KdTree<PointT>);
	pcl::EuclideanClusterExtraction<PointT> ec;
	ec.setClusterTolerance (0.04);
	ec.setMinClusterSize (100);
	ec.setMaxClusterSize (25000);
	ec.setSearchMethod (eucl_tree);

	PointCloudT::Ptr cloud_cluster;

	////////////////////
	// vfh estimate
	////////////////////
	pcl::NormalEstimation<PointT, pcl::Normal> ne;
	pcl::search::KdTree<PointT>::Ptr vfh_tree1 (new pcl::search::KdTree<PointT> ());
	pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh;
	pcl::search::KdTree<PointT>::Ptr vfh_tree2 (new pcl::search::KdTree<PointT> ());
	std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr> computed_vfhs;	
	
	ne.setSearchMethod (vfh_tree1);
	ne.setRadiusSearch (0.05);
	vfh.setSearchMethod (vfh_tree2);
	vfh.setRadiusSearch (0.05);
	pcl::PointCloud<pcl::Normal>::Ptr normals;
	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs;


	////////////////////
	// nearest neighbour
	////////////////////
	int k = 6;

	std::string kdtree_idx_file_name = "kdtree.idx";
	std::string training_data_h5_file_name = "training_data.h5";
	std::string training_data_list_file_name = "training_data.list";

//	std::vector<vfh_model> models;
//	flann::Matrix<float> data;

//	loadFileList (models, training_data_list_file_name);
//	flann::load_from_file (data, 
//						   training_data_h5_file_name, 
//						   "training_data");
//	
//	flann::Index<flann::ChiSquareDistance<float> > index (data, 
//														  flann::SavedIndexParams 
//														  ("kdtree.idx"));


    ////////////////////
	// process nearest neighbour
	////////////////////
	std::vector<hypothesis> final_hypothesis;
	final_hypothesis.clear();



	double last = pcl::getTime();

	while (! viewer->wasStopped())
	{
		if (new_cloud_available_flag)
		{

			new_cloud_available_flag = false;
			double now = pcl::getTime();

			////////////////////
			// pass filter
			////////////////////
					  
			//passed_cloud = boost::shared_ptr<PointCloudT>(new PointCloudT);

			////////////////////
			// voxel grid
			////////////////////
			//voxelized_cloud = boost::shared_ptr<PointCloudT>(new PointCloudT);

			////////////////////
			// sac segmentation
			////////////////////
	
			//cloud_f = boost::shared_ptr<PointCloudT>(new PointCloudT);	
			//cloud_plane = boost::shared_ptr<PointCloudT> (new PointCloudT);	
			//cloud_filtered = boost::shared_ptr<PointCloudT> (new PointCloudT);

			////////////////////
			// euclidean clustering
			////////////////////
			extracted_clusters.clear();
			cluster_indices.clear();

			////////////////////
			// vfh estimate
			////////////////////
			computed_vfhs.clear();


            ////////////////////
			// nearest neighbour
			////////////////////
			
			cloud_mutex.lock();
			
			//displayCloud(viewer,cloud);
			boost::thread displayCloud_(displayCloud,viewer,cloud);

			if(now-last > 13 || first_time)
			{
				first_time = false;

				last=now;

                ////////////////////
				// invert correction
				////////////////////

				pcl::transformPointCloud(*cloud,*new_cloud,transMat);
				
				////////////////////
				// pass filter
				////////////////////
				
				pass.setInputCloud (new_cloud);
				pass.setFilterFieldName ("x");
				pass.setFilterLimits (-0.5, 0.5);
				//pass.setFilterLimitsNegative (true);
				pass.filter (*passed_cloud);


				////////////////////
				// voxel grid
				////////////////////

				vg.setInputCloud (passed_cloud);
				vg.filter (*voxelized_cloud);

				////////////////////
				// sac segmentation
				////////////////////
			
				*cloud_filtered = *voxelized_cloud;

				int i=0, nr_points = (int) voxelized_cloud->points.size ();
				while (cloud_filtered->points.size () > 0.3 * nr_points)
				{
					// Segment the largest planar component from the remaining cloud
					seg.setInputCloud (cloud_filtered);
					seg.segment (*inliers, *coefficients);
					if (inliers->indices.size () == 0)
					{
						std::cout << "Couldnt estimate a planar model for the dataset.\n";
						break;
					}

					// Extract the planar inliers from the input cloud
					pcl::ExtractIndices<PointT> extract;
					extract.setInputCloud (cloud_filtered);
					extract.setIndices (inliers);
					extract.setNegative (false);

					// Get the points associated with the planar surface
					extract.filter (*cloud_plane);

					// Remove the planar inliers, extract the rest
					extract.setNegative (true);
					extract.filter (*cloud_f);
					*cloud_filtered = *cloud_f;
				}

                ////////////////////
				// euclidean clustering
				////////////////////
				
				// Creating the KdTree object for the search method of the extraction
				eucl_tree->setInputCloud (cloud_filtered);

				ec.setInputCloud (cloud_filtered);
				ec.extract (cluster_indices);

				
				for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
				{
					//PointCloudT::Ptr cloud_cluster (new PointCloudT);
					cloud_cluster = boost::shared_ptr<PointCloudT>(new PointCloudT);
					for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); pit++)
						cloud_cluster->points.push_back (cloud_filtered->points [*pit]);
					cloud_cluster->width = cloud_cluster->points.size ();
					cloud_cluster->height = 1;
					cloud_cluster->is_dense = true;
					
					extracted_clusters.push_back(cloud_cluster);
				}
				cloud_cluster.reset();

							
				////////////////////
				// vfh estimate
				////////////////////
				for (int z=0; z<extracted_clusters.size(); ++z)
				{
					vfhs = boost::shared_ptr<pcl::PointCloud<pcl::VFHSignature308> > (new pcl::PointCloud<pcl::VFHSignature308>);
					normals = boost::shared_ptr<pcl::PointCloud<pcl::Normal> > (new pcl::PointCloud<pcl::Normal>);

					ne.setInputCloud (extracted_clusters.at(z));
					ne.compute (*normals);
					vfh.setInputCloud (extracted_clusters.at(z));
					vfh.setInputNormals (normals);
					vfh.compute (*vfhs);
					computed_vfhs.push_back(vfhs);
	
				}
				vfhs.reset();
				normals.reset();

				////////////////////
				// nearest neighbour
				////////////////////	

				std::vector<vfh_model> models;
				flann::Matrix<int> k_indices;
				flann::Matrix<float> k_distances;
				flann::Matrix<float> data;


				loadFileList (models, training_data_list_file_name);
				flann::load_from_file (data, 
									   training_data_h5_file_name, 
									   "training_data");
	
				flann::Index<flann::ChiSquareDistance<float> > index 
					(data, 
					 flann::SavedIndexParams 
					 ("kdtree.idx"));


				for(int z=0; z<computed_vfhs.size(); ++z)
				{
					vfh_model histogram;
					histogram.second.resize(308);
	
					for (size_t i = 0; i < 308; ++i)
					{
						histogram.second[i] = computed_vfhs.at(z)->points[0].histogram[i];
					}

					index.buildIndex ();
					nearestKSearch (index, histogram, k, k_indices, k_distances);
					
					hypothesis x;
					x.distance = k_distances[0][0];
					size_t index = models.at(k_indices[0][0]).first.find("_v",5);

					x.object_name = models.at(k_indices[0][0]).first.substr(5,index-5);

					ddd = boost::shared_ptr<PointCloudT>(new PointCloudT);
					pcl::transformPointCloud(*extracted_clusters.at(z),*ddd,transMat);
					x.cluster = ddd;
					ddd.reset();

					std::string filename ="";
					filename += "inputcloud_" + boost::lexical_cast<std::string>(j+1);
					filename += "_" + boost::lexical_cast<std::string>(z) + ".pcd";
					x.cluster_name = filename.c_str();

					final_hypothesis.push_back(x);

					x.cluster.reset();
					//delete x;
					
//                    std::string filename ="";
//					filename += "inputcloud_" + boost::lexical_cast<std::string>(j+1);
//					filename += "_" + boost::lexical_cast<std::string>(z) + ".pcd";
//					const char* filen = filename.c_str();
//					fprintf(objects,"%s",filen);
//					fprintf(objects,"::");
//					fprintf(objects,models.at (k_indices[0][0]).first.c_str());
//					fprintf(objects,"::");
//					fprintf(objects,"%f",k_distances[0][0]);
//					fprintf(objects,"\n");					
				}				
				delete[] k_indices.ptr ();
				delete[] k_distances.ptr ();
				delete[] data.ptr ();
				
				std::cout << final_hypothesis.size() << std::endl;
				
				viewer->removeAllShapes();

				for(int z=0; z<final_hypothesis.size();++z)
				{
					if(final_hypothesis.at(z).distance < 100)
					{
						fprintf(objects,"%s",final_hypothesis.at(z).cluster_name.c_str());
						fprintf(objects,"::");
						fprintf(objects,"%s",final_hypothesis.at(z).object_name.c_str());
						fprintf(objects,"::");
						fprintf(objects,"%f",final_hypothesis.at(z).distance);
						fprintf(objects,"\n");
					std::stringstream ddd;
						ddd << final_hypothesis.at(z).object_name;
						ddd << "\n" << "(";
						ddd << final_hypothesis.at(z).distance;
						ddd << ")";

						viewer->addText3D(ddd.str().c_str(),
										 final_hypothesis.at(z).cluster->points[0],
										 0.02,1,1,1,
										 boost::lexical_cast<std::string>(z));
						drawBoundingBox(final_hypothesis.at(z).cluster,viewer,z);
					}
				}
				//boost::thread allBoxes_(allBoxes,viewer,final_hypothesis);		
				//allBoxes_.join();
				viewer->spinOnce();
				final_hypothesis.clear();
				j++;
			}

//			for(int z=0; z<extracted_clusters.size(); ++z)
//			{
//				//viewer->addPointCloud<PointT>(extracted_clusters.at(z),
//				//							 boost::lexical_cast<std::string>(z));
//
//				std::string filename ="";
//				filename += "inputcloud_" + boost::lexical_cast<std::string>(j);
//				filename += "_" + boost::lexical_cast<std::string>(z) + ".pcd";
//				pcl::io::savePCDFile(filename,*extracted_clusters.at(z),false);
//			}	


//			for(int z=0; z<computed_vfhs.size(); ++z)
//			{
//				//viewer->addPointCloud<PointT>(extracted_clusters.at(z),
//				//							 boost::lexical_cast<std::string>(z));
//
//				std::string filename ="";
//				filename += "inputcloud_" + boost::lexical_cast<std::string>(j);
//				filename += "_" + boost::lexical_cast<std::string>(z) + "_vfhs.pcd";
//				pcl::io::savePCDFileASCII<pcl::VFHSignature308> (filename, *computed_vfhs.at(z));
//			}			


			//viewer->removeAllShapes();
//			viewer->removeAllPointClouds();
//			viewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0); 
//			pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
//			viewer->addPointCloud<PointT>(cloud,rgb,"input_cloud");
//			viewer->spinOnce();
		

			//std::cout << final_hypothesis.at(0).cluster->points[0];

			//boost::this_thread::sleep(boost::posix_time::milliseconds(10));
			displayCloud_.join();
			cloud_mutex.unlock();
		}
    }
	grab->stop();
	return 0;
}
