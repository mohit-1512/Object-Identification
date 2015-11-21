#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>
#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
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

class Bazzinga
{
public:
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

	Bazzinga(float leaf_size) :
		viewer_ ("BAZZINGA"),
		first_time_ (false),
		new_cloud_available_flag_ (false),
		leaf_size_ (leaf_size),
		j_ (0),
		k_ (6)
								   
		{
			// bounding box
			p2w_ = Eigen::Matrix4f::Identity();

			// basic cloud
			cloud_ = boost::shared_ptr<PointCloudT>(new PointCloudT);
			new_cloud_ = boost::shared_ptr<PointCloudT>(new PointCloudT);

			// open file
			ddd_ = boost::shared_ptr<PointCloudT>(new PointCloudT);
			objects_ = fopen ("objects.txt","a");			

			
			// pass filter
			transMat_ = Eigen::Matrix4f::Identity(); 
			transMat_(1,1) = -1;

			// pass filter
			passed_cloud_ = boost::shared_ptr<PointCloudT>(new PointCloudT);
			
			// voxel
			voxelized_cloud_ = boost::shared_ptr<PointCloudT>(new PointCloudT);
			vg_.setLeafSize (leaf_size_, leaf_size_, leaf_size_);
			
			// sac segmentation
			cloud_f_ = boost::shared_ptr<PointCloudT>(new PointCloudT);	
			cloud_plane_ = boost::shared_ptr<PointCloudT> (new PointCloudT);	
			cloud_filtered_ = boost::shared_ptr<PointCloudT> (new PointCloudT);

			//in cloud_cb
			//inliers = boost::shared_ptr<pcl::PointIndices>(new pcl::PointIndices);
			//coefficients = boost::shared_ptr<pcl::ModelCoefficients>
			//(new pcl::ModelCoefficients);

			seg_.setOptimizeCoefficients (true);
			seg_.setModelType (pcl::SACMODEL_PLANE);
			seg_.setMethodType (pcl::SAC_RANSAC);
			seg_.setMaxIterations (100);
			seg_.setDistanceThreshold (0.02);

			// euclidean
			eucl_tree_ = boost::shared_ptr<pcl::search::KdTree<PointT> >
				(new pcl::search::KdTree<PointT>);


			// vfh estimate
			vfh_tree1_ = boost::shared_ptr<pcl::search::KdTree<PointT> >
				(new pcl::search::KdTree<PointT>);
			vfh_tree2_ = boost::shared_ptr<pcl::search::KdTree<PointT> >
				(new pcl::search::KdTree<PointT>);
			ne_.setSearchMethod (vfh_tree1_);
			ne_.setRadiusSearch (0.05);
			vfh_.setSearchMethod (vfh_tree2_);
			vfh_.setRadiusSearch (0.05);

			// nearest neighbour
			kdtree_idx_file_name_ = "kdtree.idx";
			training_data_h5_file_name_ = "training_data.h5";
			training_data_list_file_name_ = "training_data.list";
			
			loadFileList (models_, training_data_list_file_name_);
			flann::load_from_file (data_, 
								   training_data_h5_file_name_, 
								   "training_data");

			flann::Index<flann::ChiSquareDistance<float> > index_
				(data_, 
				 flann::SavedIndexParams 
				 ("kdtree.idx"));
			index_.buildIndex ();	

			last_ = pcl::getTime();
		}


	pcl::visualization::CloudViewer viewer_;
	boost::mutex cloud_mutex_;

	int j_;
	float leaf_size_;
	
	// bounding box
	Eigen::Vector4f centroid_;
	Eigen::Matrix3f covariance_;
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_;
	Eigen::Matrix3f eigDx_;	
	Eigen::Matrix4f p2w_;
	pcl::PointCloud<PointT> cPoints_;
	PointT min_pt_, max_pt_;
	
	PointCloudT::Ptr cloud_;
	PointCloudT::Ptr new_cloud_;
	bool new_cloud_available_flag_;

	PointCloudT::Ptr ddd_;
	bool first_time_;
	FILE* objects_;

	// inversion
	Eigen::Matrix4f transMat_;
	
	// voxel
	PointCloudT::Ptr passed_cloud_;
	pcl::PassThrough<PointT> pass_;
	PointCloudT::Ptr voxelized_cloud_;
	pcl::VoxelGrid<PointT> vg_;
	
	// sac segmentation
	PointCloudT::Ptr cloud_f_;
	PointCloudT::Ptr cloud_plane_;
	PointCloudT::Ptr cloud_filtered_;
	pcl::SACSegmentation<PointT> seg_;

	//in cloud_cb
	//pcl::PointIndices::Ptr inliers;
	//pcl::ModelCoefficients::Ptr coefficients;

	// euclidean clustering
	//std::vector<pcl::PointIndices> cluster_indices_;
	std::vector<PointCloudT::Ptr> extracted_clusters_;
	pcl::search::KdTree<PointT>::Ptr eucl_tree_;
	pcl::EuclideanClusterExtraction<PointT> ec_;

	PointCloudT::Ptr cloud_cluster_;


	// vfh estimate
	pcl::NormalEstimation<PointT, pcl::Normal> ne_;
	pcl::search::KdTree<PointT>::Ptr vfh_tree1_;
	pcl::VFHEstimation<PointT, pcl::Normal, pcl::VFHSignature308> vfh_;
	pcl::search::KdTree<PointT>::Ptr vfh_tree2_;
	std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr> computed_vfhs_;
	
	pcl::PointCloud<pcl::Normal>::Ptr normals_;
	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs_;
	

	// nearest neighbour
	int k_;
	std::string kdtree_idx_file_name_;
	std::string training_data_h5_file_name_;
	std::string training_data_list_file_name_;

	std::vector<vfh_model> models_;
	flann::Matrix<int> k_indices_;
	flann::Matrix<float> k_distances_;
	flann::Matrix<float> data_;

	//flann::Index<flann::ChiSquareDistance<float> > index_;


	PointCloudT::Ptr ddd;	


	// final
	std::vector<hypothesis> final_hypothesis_;


	double last_;
	double now_;
	


	void
	passFilter(PointCloudT::Ptr &cloud,
			   PointCloudT::Ptr &target_cloud)
	{
		pass_.setInputCloud (cloud);
		pass_.setFilterFieldName ("x");
		pass_.setFilterLimits (-0.5, 0.5);
		pass_.filter (*target_cloud);
	}
	
	void
	voxel(PointCloudT::Ptr &cloud,
		  PointCloudT::Ptr &target_cloud)
		{
			vg_.setInputCloud (cloud);
			vg_.filter (*target_cloud);
		}

	

	void
	planeSegment(PointCloudT::Ptr &cloud,
				 pcl::ModelCoefficients::Ptr &coefficients,
				 pcl::PointIndices::Ptr &inliers)
		{
			//cloud_filtered_ = voxelized_cloud_;
			int i=0, nr_points = (int) voxelized_cloud_->points.size ();
			while (voxelized_cloud_->points.size () > 0.3 * nr_points)
			{
				// Segment the largest planar component from the remaining cloud
				seg_.setInputCloud (voxelized_cloud_);
				seg_.segment (*inliers, *coefficients);
				if (inliers->indices.size () == 0)
				{
					std::cout << "Couldnt estimate a planar model for the dataset.\n";
					break;
				}

				// Extract the planar inliers from the input cloud
				pcl::ExtractIndices<PointT> extract;
				extract.setInputCloud (voxelized_cloud_);
				extract.setIndices (inliers);
				extract.setNegative (false);

				// Get the points associated with the planar surface
				extract.filter (*cloud_plane_);

				// Remove the planar inliers, extract the rest
				extract.setNegative (true);
				extract.filter (*cloud_f_);
				*voxelized_cloud_ = *cloud_f_;
			}
		}
	
	void
	euclideanSegment(const PointCloudT::ConstPtr &cloud,
					 std::vector<pcl::PointIndices> &cluster_indices)
		{
			ec_.setClusterTolerance (0.04);
			ec_.setMinClusterSize (100);
			ec_.setMaxClusterSize (25000);
			ec_.setSearchMethod (eucl_tree_);

			eucl_tree_->setInputCloud (cloud);
			ec_.setInputCloud (cloud);
			ec_.extract (cluster_indices);
		}

	void
	extractCluster(PointCloudT::Ptr& cloud,
				   std::vector<pcl::PointIndices> &cluster_indices,
				   std::vector<PointCloudT::Ptr> &extracted_clusters)
		{
			extracted_clusters.clear();

			for (std::vector<pcl::PointIndices>::const_iterator it = 
					 cluster_indices.begin (); it != cluster_indices.end (); ++it)
			{
				cloud_cluster_.reset(new PointCloudT);
				
				for (std::vector<int>::const_iterator pit = it->indices.begin (); 
					 pit != it->indices.end (); pit++)
					cloud_cluster_->points.push_back (cloud->points [*pit]);
				cloud_cluster_->width = cloud_cluster_->points.size ();
				cloud_cluster_->height = 1;
				cloud_cluster_->is_dense = true;
					
				extracted_clusters.push_back(cloud_cluster_);
			}
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
			flann::Matrix<float> p = flann::Matrix<float>
				(new float[model.second.size ()], 1, model.second.size ());
			memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

			indices = flann::Matrix<int>(new int[k], 1, k);
			distances = flann::Matrix<float>(new float[k], 1, k);
			index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
			delete[] p.ptr ();
		}

	void
	drawBoundingBox(std::vector<PointCloudT::Ptr> &extracted_clusters,
					pcl::visualization::PCLVisualizer& viz)
		{
			
			for(int z=0; z<5; ++z)
			{

				pcl::compute3DCentroid(*extracted_clusters.at(z), centroid_);
				computeCovarianceMatrixNormalized(*extracted_clusters.at(z), 
												  centroid_, covariance_);

				eigen_solver_.compute(covariance_, Eigen::ComputeEigenvectors);
				eigDx_ = eigen_solver_.eigenvectors();
				eigDx_.col(2) = eigDx_.col(0).cross(eigDx_.col(1));

				p2w_.block<3,3>(0,0) = eigDx_.transpose();
				p2w_.block<3,1>(0,3) = -1.f * (p2w_.block<3,3>(0,0) * centroid_.head<3>());
				pcl::transformPointCloud(*extracted_clusters.at(z), cPoints_, p2w_);


				pcl::getMinMax3D(cPoints_, min_pt_, max_pt_);
				const Eigen::Vector3f mean_diag = 0.5f * (max_pt_.getVector3fMap() 
														  + min_pt_.getVector3fMap());

				const Eigen::Quaternionf qfinal(eigDx_);
				const Eigen::Vector3f tfinal = eigDx_*mean_diag + centroid_.head<3>();
   
				viz.removeShape(boost::lexical_cast<std::string>(z));
				viz.addCube(tfinal, qfinal, max_pt_.x - min_pt_.x, max_pt_.y - min_pt_.y, 
							max_pt_.z - min_pt_.z,
							boost::lexical_cast<std::string>(z));
			}
		}


	void
	calcVFHS (std::vector<PointCloudT::Ptr>& extracted_clusters,
			  pcl::PointCloud<pcl::Normal>::Ptr& normals,
			  pcl::PointCloud<pcl::VFHSignature308>::Ptr& vfhs,
			  std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr>& computed_vfhs)
		{
			computed_vfhs_.clear();
		
			for (int z=0; z<extracted_clusters.size(); ++z)
			{
				vfhs_ = boost::shared_ptr<pcl::PointCloud<pcl::VFHSignature308> > 
					(new pcl::PointCloud<pcl::VFHSignature308>);
				normals_ = boost::shared_ptr<pcl::PointCloud<pcl::Normal> > 
					(new pcl::PointCloud<pcl::Normal>);

				ne_.setInputCloud (extracted_clusters.at(z));
				ne_.compute (*normals);
				vfh_.setInputCloud (extracted_clusters.at(z));
				vfh_.setInputNormals (normals);
				vfh_.compute (*vfhs);
				computed_vfhs.push_back(vfhs);
	
			}

			vfhs.reset(new pcl::PointCloud<pcl::VFHSignature308>);
			normals.reset(new pcl::PointCloud<pcl::Normal>);
		}


	void
	calcNN (std::vector<PointCloudT::Ptr>& extracted_clusters,
			std::vector<pcl::PointCloud<pcl::VFHSignature308>::Ptr>& computed_vfhs,
			std::vector<hypothesis>& final_hypothesis,
			flann::Index<flann::ChiSquareDistance<float> > &index)
		{
			final_hypothesis.clear();

			for(int z=0; z<computed_vfhs.size(); ++z)
			{
				vfh_model histogram;
				histogram.second.resize(308);
	
				for (size_t i = 0; i < 308; ++i)
				{
					histogram.second[i] = computed_vfhs.at(z)->points[0].histogram[i];
				}

				nearestKSearch (index, histogram, k_, k_indices_, k_distances_);
					
				hypothesis x;
				x.distance = k_distances_[0][0];
				size_t index = models_.at(k_indices_[0][0]).first.find("_v",5);

				x.object_name = models_.at(k_indices_[0][0]).first.substr(5,index-5);

				ddd = boost::shared_ptr<PointCloudT>(new PointCloudT);
				pcl::transformPointCloud(*extracted_clusters.at(z),*ddd,transMat_);
				x.cluster = ddd;
				ddd.reset();

				std::string filename ="";
				filename += "inputcloud_";
				filename += "_" + boost::lexical_cast<std::string>(z) + ".pcd";
				x.cluster_name = filename.c_str();

				final_hypothesis.push_back(x);

				x.cluster.reset();
			}

			delete[] k_indices_.ptr ();
			delete[] k_distances_.ptr ();
			//delete[] data.ptr ();			
		}
	

	void 
	viz_cb(pcl::visualization::PCLVisualizer& viz)
		{
			boost::mutex::scoped_lock lock(cloud_mutex_);

			pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud_);
			//viz.setCameraPosition (0,0,-1,0,0,5,0,1,0);

			viz.resetCamera();
			if(!new_cloud_)
			{
				boost::this_thread::sleep (boost::posix_time::seconds(1));
				return;
			}
			

			if (!viz.updatePointCloud (voxelized_cloud_, "input"))
			{
				viz.addPointCloud (voxelized_cloud_, rgb, "input");
				viz.resetCameraViewpoint ("input");
			}

			if(new_cloud_available_flag_)
			{
				drawBoundingBox(extracted_clusters_,viz);
				
				viz.removeShape("clusters");
				viz.addText("Clusters:",10,120,20,1.0,1.0,1.0,"clusters");

				viz.removeShape("#clusters");
				viz.addText(boost::lexical_cast<std::string>(extracted_clusters_.size()),
							100,120,20,1.0,1.0,1.0,"#clusters");

			}

			new_cloud_available_flag_ = false;
			
		}

	void
	saveFile(pcl::PointCloud<pcl::VFHSignature308>::Ptr &cloud,
			 int j)
		{
			std::string filename ="";
			filename += "inputcloud_" + boost::lexical_cast<std::string>(j);
			filename += "_"; 
			filename += ".pcd";
			pcl::io::savePCDFileASCII<pcl::VFHSignature308> (filename,*cloud);
		}


	void 
	cloud_cb(const PointCloudT::ConstPtr& cloud)
		{
			boost::mutex::scoped_lock lock(cloud_mutex_);
			pcl::transformPointCloud(*cloud,*new_cloud_,transMat_);
			passFilter(new_cloud_,passed_cloud_);
			voxel(passed_cloud_,voxelized_cloud_);
			
			now_ = pcl::getTime();
			
			//if (now_ - last_ > 5)
			//{
				first_time_ = false;
				last_ = now_;

				pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
				pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
				planeSegment(voxelized_cloud_, coefficients, inliers);
			
				std::vector<pcl::PointIndices> cluster_indices;
				cluster_indices.clear();
				
				euclideanSegment(voxelized_cloud_, cluster_indices);

				extractCluster(voxelized_cloud_,cluster_indices,extracted_clusters_);				
				//calcVFHS(extracted_clusters_,normals_,vfhs_,computed_vfhs_);

				//calcNN (extracted_clusters_,computed_vfhs_,final_hypothesis_,index);

				//for(int z=0; z<computed_vfhs_.size(); ++z)
				//saveFile(computed_vfhs_.at(z),z);
				
				//boost::this_thread::sleep (boost::posix_time::seconds (2));				
				//}
			new_cloud_available_flag_ = true;
		}


	void
	run ()
		{
			pcl::Grabber* interface = new pcl::OpenNIGrabber ();
			boost::function<void (const PointCloudT::ConstPtr&)> f =
				boost::bind (&Bazzinga::cloud_cb, this, _1);
			interface->registerCallback (f);
    
			viewer_.runOnVisualizationThread 
			(boost::bind(&Bazzinga::viz_cb, this, _1), "viz_cb");
    
			interface->start ();
  
			while (!viewer_.wasStopped ())
				boost::this_thread::sleep(boost::posix_time::seconds(1));
			interface->stop ();
		}
  
};


int main (int argc, char** argv)
{
	Bazzinga x (0.001);
	x.run();
}

