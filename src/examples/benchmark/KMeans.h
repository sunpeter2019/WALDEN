#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

class Point{

private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

public:
    Point(int id, double* line,int n_features){
        dimensions = 0;
        pointId = id;
		for (int i = 0; i < n_features; i++){
			values.push_back(line[i]);
			dimensions++;
		}
        clusterId = 0; //Initially not assigned to any cluster
    }

    int getDimensions(){
        return dimensions;
    }

    int getCluster(){
        return clusterId;
    }

    int getID(){
        return pointId;
    }

    void setCluster(int val){
        clusterId = val;
    }

    double getVal(int pos){
        return values[pos];
    }
};

class Cluster{

private:
    int clusterId;
    vector<double> centroid;
    vector<Point> points;

public:
    Cluster(int clusterId, Point centroid){
        this->clusterId = clusterId;
        for(int i=0; i<centroid.getDimensions(); i++){
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(centroid);
    }

    void addPoint(Point p){
        p.setCluster(this->clusterId);
        points.push_back(p);
    }

    bool removePoint(int pointId){
        int size = points.size();

        for(int i = 0; i < size; i++)
        {
            if(points[i].getID() == pointId)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    int getId(){
        return clusterId;
    }

    Point getPoint(int pos){
        return points[pos];
    }

    int getSize(){
        return points.size();
    }

    double getCentroidByPos(int pos) {
        return centroid[pos];
    }

    void setCentroidByPos(int pos, double val){
        this->centroid[pos] = val;
    }
};

class customRND
{
public:
    void seed( unsigned int s ) { _seed = s; }
    customRND() : _seed( 0 ), _a( 214013/*1103515245*/ ), _c( 2531011/*12345*/ ), _m( 2147483648 ) {}
    int rnd() { return( _seed = ( _a * _seed + _c ) % _m ); }
 
    int _a, _c;
    unsigned int _m, _seed;
};

class KMeans{
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
	customRND myRND;
	
    int getNearestClusterId(Point point){
        double sum = 0.0, min_dist;
        int NearestClusterId;

        for(int i = 0; i < dimensions; i++)
        {
            sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
        }

        min_dist = sqrt(sum);
        NearestClusterId = clusters[0].getId();

        for(int i = 1; i < K; i++)
        {
            double dist;
            sum = 0.0;

            for(int j = 0; j < dimensions; j++)
            {
                sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
            }

            dist = sqrt(sum);

            if(dist < min_dist)
            {
                min_dist = dist;
                NearestClusterId = clusters[i].getId();
            }
        }

        return NearestClusterId;
    }

public:
    KMeans(int K, int iterations){
        this->K = K;
        this->iters = iterations;
    }

    int* run(vector<Point>& all_points,int*labels, unsigned int rndseed=0){
		
		this->myRND.seed(rndseed);
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

		//Initializing Clusters
        vector<int> used_pointIds;

        for(int i=1; i<=K; i++)
        {
            while(true)
            {
                //int index = rand() % total_points;
				int index = myRND.rnd() % total_points;

                if(find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end())
                {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    Cluster cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
		
        int iter = 1;
        while(true)
        {
            bool done = true;

            // Add all points to their nearest cluster
            for(int i = 0; i < total_points; i++)
            {
                int currentClusterId = all_points[i].getCluster();
                int nearestClusterId = getNearestClusterId(all_points[i]);

                if(currentClusterId != nearestClusterId)
                {
                    if(currentClusterId != 0){
                        for(int j=0; j<K; j++){
                            if(clusters[j].getId() == currentClusterId){
                                clusters[j].removePoint(all_points[i].getID());
                            }
                        }
                    }

                    for(int j=0; j<K; j++){
                        if(clusters[j].getId() == nearestClusterId){
                            clusters[j].addPoint(all_points[i]);
                        }
                    }
                    all_points[i].setCluster(nearestClusterId);
                    done = false;
                }
				labels[i] = currentClusterId-1;
            }
            if(done || iter >= iters)
                break;
            iter++;
        }
		return labels;
    }
};
void getKMeansLabels(int K,double** X,int* label,int iters,int n_samples,int n_features, unsigned int rndseed=0){
	int pointId = 1;
    vector<Point> all_points;
	for (int i = 0; i < n_samples; i++){
		Point point(pointId, X[i], n_features);
        all_points.push_back(point);
        pointId++;
	}
	if(all_points.size() < K){
        cout<<"Error: Number of clusters greater than number of points."<<endl;
		exit(0);
    }
	KMeans kmeans(K, iters);
	kmeans.run(all_points,label, rndseed);
}