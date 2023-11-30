// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "zipf.h"
#include "GaussianMixture.h"
#include <cmath>
#include <random>
#include <unordered_set>

template <class T>
bool load_binary_data(T data[], int length, const std::string& file_path) {
  std::ifstream is(file_path.c_str(), std::ios::binary | std::ios::in);
  if (!is.is_open()) {
    return false;
  }
  is.read(reinterpret_cast<char*>(data), std::streamsize(length * sizeof(T)));
  is.close();
  return true;
}

template <class T>
bool load_text_data(T array[], int length, const std::string& file_path) {
  std::ifstream is(file_path.c_str());
  if (!is.is_open()) {
    return false;
  }
  int i = 0;
  std::string str;
  while (std::getline(is, str) && i < length) {
    std::istringstream ss(str);
    ss >> array[i];
    i++;
  }
  is.close();
  return true;
}

template <class T>
T* get_search_keys(T array[],int start, int num_keys, int num_searches, std::unordered_set<T> ab) {
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<int> dis(start, num_keys - 1);
  auto* keys = new T[num_searches];
  for (int i = 0; i < num_searches; i++) {
    int pos = dis(gen);
    if(ab.find(array[pos]) != ab.end() ){
      keys[i] = array[pos];
    }
    else{
      i--;
    }
  }
  return keys;
}

template <class T>
T* get_search_keys(T array[], int num_keys, int num_searches) {
  std::mt19937_64 gen(std::random_device{}());
  std::uniform_int_distribution<int> dis(0, num_keys - 1);
  auto* keys = new T[num_searches];
  for (int i = 0; i < num_searches; i++) {
    int pos = dis(gen);    
    keys[i] = array[pos];    
  }
  return keys;
}


template <class T>
T* get_search_keys_zipf(T array[], int num_keys, int num_searches) {
  auto* keys = new T[num_searches];
  ScrambledZipfianGenerator zipf_gen(num_keys);
  for (int i = 0; i < num_searches; i++) {
    int pos = zipf_gen.nextValue();
    keys[i] = array[pos];
  }
  return keys;
}



double gauss3(double x, double a1, double m1,  double s1, double a2,  double m2,double s2, double a3,  double m3,  double s3) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3

    return a1 * expf(-(x - m1) * (x - m1) / (2. * s1 * s1)) + a2 * expf(-(x - m2) * (x - m2) / (2. * s2 * s2)) + a3 * expf(-(x - m3) * (x - m3) / (2. * s3 *s3));
        
}

double gauss4(double x, double a1, double m1,  double s1, double a2,  double m2,double s2, double a3,  double m3,  double s3, double a4,  double m4,  double s4) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3

    return a1 * expf(-(x - m1) * (x - m1) / (2. * s1 * s1)) + a2 * expf(-(x - m2) * (x - m2) / (2. * s2 * s2)) + a3 * expf(-(x - m3) * (x - m3) / (2. * s3 *s3))+ a4 * expf(-(x - m4) * (x - m4) / (2. * s4 *s4));
        
}

double gauss5(double x, double a1, double m1,  double s1, double a2,  double m2,double s2, double a3,  double m3,  double s3, double a4,  double m4,  double s4, double a5,  double m5,  double s5) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3

    return a1 * expf(-(x - m1) * (x - m1) / (2. * s1 * s1)) + a2 * expf(-(x - m2) * (x - m2) / (2. * s2 * s2)) + a3 * expf(-(x - m3) * (x - m3) / (2. * s3 *s3))+ a4 * expf(-(x - m4) * (x - m4) / (2. * s4 *s4))+ a5 * expf(-(x - m5) * (x - m5) / (2. * s5 *s5));
        
}

// you can save your dataset distribution here
double gen_weight_data(double key, int weight_file_type ) {

    double Weight = 0;

    switch (weight_file_type)
    {
        case 1 :
            Weight = gauss4(key, 3.53436723e-02 ,-2.13950974e+04, 8.57312408e+02, 7.02669126e-02, -1.55899792e+04, 2.14474169e+03, 4.37179796e-02, 1.78329011e+03, 1.36272099e+03,  0.00598831229, 25000.52509306,   2185.99989178); //longlat-around6.bin-1
            break;
        case 2 :
            Weight = gauss3(key, 3.53436723e-02 ,-2.13950974e+04, 8.57312408e+02, 7.02669126e-02, -1.55899792e+04, 2.14474169e+03, 4.37179796e-02, 1.78329011e+03, 1.36272099e+03); //longlat-around6.bin-2
            break;
        case 3 :
            Weight = gauss3(key, 0.1001731,  35.55802242 , 7.86110845, 0.05720341 ,54.60873921  ,7.00619841, 0,0,0); //gowalla-around6.bin-3
            break;
        case 4 :
            Weight = gauss3(key, 2.41085344e-02, -1.13735899e+02, -9.88618360e+00, 5.13968145e-02, -8.22674611e+01,  9.63536770e+00, 0.0656748,  9.46394118, 8.28807226); //longitudes-around6.bin-4
            break;
        case 5 :
            Weight = gauss4(key, 4.11727682e-01, 5.07920902e+18, 1.12286010e+17, 6.52006327e-02, 1.21068408e+18, 1.87047896e+17, 1.30720403e-01, 9.86318293e+18, 1.22665478e+17, 4.82435159e-02, 5.85443351e+18, 2.51042365e+17); //osm-around8.bin-5
            break;
        case 6 :
            Weight = gauss3(key, 2.91199927e-02, -1.39306307e+18,  3.60808154e+18, 0 , 0  , 0, 0,0,0); //books.bin-6
            break;
        case 7 :
            Weight = gauss3(key,2.91108180e-06, -1.39055148e+18,  3.60691274e+18, 0 , 0  , 0, 0,0,0); //books.bin round5
            break;
        case 8 :
            std::cout << "You entered wrong dataset number\n";
            break;
        default:
            std::cout << "You did not enter dataset number!\n";
    }
    return Weight;
}



template <class T>
T* get_search_keys_normal(T array[], int num_keys, int num_searches) {
    auto* keys = new T[num_searches];    

    int mu = num_keys / 2;
    float sigma = num_keys / 1000.0f;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mu, sigma);

    //double number = distribution(generator);
    

    for (int i = 0; i < num_searches; i++) {
        
        int pos = (int)distribution(generator);
        if (pos >= num_keys)
            pos = num_keys - 1;
        keys[i] = array[pos];
        
    }
    
    return keys;
}

//template <class T>
double gen_weight(int keyrank, int length) {

    const float two_pi = 2.0 * 3.14159265358979323846;

    int mu = length / 2;
    float sigma = length /1000.0;
    float sigma22 = 2 * sigma * sigma;
    float sigma_sqrt2PI = sqrtf(two_pi) * sigma;   //https://www.cnblogs.com/mightycode/p/8370616.html

    double Weight = (expf(-(keyrank - mu) * (keyrank - mu) / sigma22)) / sigma_sqrt2PI;

    return Weight;
}

template <class T>
double* gen_fit(T keys[], int number_data) {
  //const int number_data = 20000000; //The number of data points
	const int number_features = 1; //The number of features in the data array
	//int nb_clusters = 3; //The expected number of clusters
	int max_iterations = 1000; //The maximum number of iterations
	string type_covariance = "full"; //The GMM covariance type. The current supported option is "full" only.
	int rndseed = 21; //The seed for the random generator. Fixing this seed prevents a potential convergeance to different clustering results upon different runs
	GaussianMixture *gmm;
	
	//The data matrix
	double **data = new double  * [number_data];
	for(int i=0; i<number_data; i++){
		data[i] = new double [number_features];
	}
	
	// std::string keys_file_path =  "../data/longlat-around4-shuffle.bin";
	// auto keys = new double[number_data];
  // load_binary_data(keys, number_data, keys_file_path);

	for(int i=0; i<number_data; i++){
		for(int j=0; j<number_features; j++){
			data[i][j] = keys[i];
			//cout<<"\t"<<data[i][j];
		}
		//cout<<endl; 
	}
    //double arr[number_data][1]=keys;
	// typedef double (*pinta)[number_features]; //A 2D array of columns of 2
    // pinta arr = (pinta)keys;

	double lbic=0;
	int lcluster=0;
	double*weights; double**means; double***covariances;
	for(int cluster=1;cluster<5;cluster++){
		gmm = new GaussianMixture(cluster,number_data, number_features, type_covariance,max_iterations,0.01,0.001,rndseed);
		//The resulting labels vector
		int *p = new int[number_data];
		//Training the the model
		(*gmm).train(data,p,number_data,number_features);
		if(lbic ==0){
			lbic = (*gmm).bic();
			lcluster=1;
			(*gmm).get_parameters_full_(weights,means,covariances);
		}
		else{
			if (lbic>(*gmm).bic() )
			{
				lbic = (*gmm).bic();
				lcluster=cluster;
				(*gmm).get_parameters_full_(weights,means,covariances);
			}			
		}		
	}
	
	auto para = new double[1+3*lcluster];
	para[0] = lcluster;
	int temp =1;


	for(int i=0;i<lcluster;i++){

		para[temp++] = weights[i];
	}

	for(int i=0; i<lcluster; i++){
		for(int j=0; j<number_features; j++){			
			
			para[temp++] = means[i][j];
		}
		
	}
	

	//cout<<"covariance:\n";	
	for(int i=0; i<lcluster; i++){
		for(int j=0; j<number_features; j++){
			for(int k=0; k<number_features; k++){							
				para[temp++] = covariances[i][j][k];
			}
		}
	}

  return para;
}

template <class T>
double* gen_fit(vector<T> keys, int number_data) {
  //const int number_data = 20000000; //The number of data points
	const int number_features = 1; //The number of features in the data array
	//int nb_clusters = 3; //The expected number of clusters
	int max_iterations = 1000; //The maximum number of iterations
	string type_covariance = "full"; //The GMM covariance type. The current supported option is "full" only.
	int rndseed = 21; //The seed for the random generator. Fixing this seed prevents a potential convergeance to different clustering results upon different runs
	GaussianMixture *gmm;
	
	//The data matrix
	double **data = new double  * [number_data];
	for(int i=0; i<number_data; i++){
		data[i] = new double [number_features];
	}
	
	// std::string keys_file_path =  "../data/longlat-around4-shuffle.bin";
	// auto keys = new double[number_data];
  // load_binary_data(keys, number_data, keys_file_path);

	for(int i=0; i<number_data; i++){
		for(int j=0; j<number_features; j++){
			data[i][j] = keys[i];
			//cout<<"\t"<<data[i][j];
		}
		//cout<<endl; 
	}
    //double arr[number_data][1]=keys;
	// typedef double (*pinta)[number_features]; //A 2D array of columns of 2
    // pinta arr = (pinta)keys;

	double lbic=0;
	int lcluster=0;
	double*weights; double**means; double***covariances;
	for(int cluster=1;cluster<5;cluster++){
		gmm = new GaussianMixture(cluster,number_data, number_features, type_covariance,max_iterations,0.01,0.001,rndseed);
		//The resulting labels vector
		int *p = new int[number_data];
		//Training the the model
		(*gmm).train(data,p,number_data,number_features);
		if(lbic ==0){
			lbic = (*gmm).bic();
			lcluster=1;
			(*gmm).get_parameters_full_(weights,means,covariances);
		}
		else{
			if (lbic>(*gmm).bic() )
			{
				lbic = (*gmm).bic();
				lcluster=cluster;
				(*gmm).get_parameters_full_(weights,means,covariances);
			}			
		}		
	}
	
	auto para = new double[1+3*lcluster];
	para[0] = lcluster;
	int temp =1;

	//cout<<"weight:\n";
	for(int i=0;i<lcluster;i++){
		//cout<<"\t"<<weights[i];
		para[temp++] = weights[i];
	}
	//cout<<endl;

	//cout<<"mean:\n";
	for(int i=0; i<lcluster; i++){
		for(int j=0; j<number_features; j++){			
			//cout<<"\t"<<means[i][j];
			para[temp++] = means[i][j];
		}
		//cout<<endl;
	}
	//cout<<endl;

	//cout<<"covariance:\n";	
	for(int i=0; i<lcluster; i++){
		for(int j=0; j<number_features; j++){
			for(int k=0; k<number_features; k++){			
				//cout<<"\t"<<covariances[i][j][k];
				para[temp++] = covariances[i][j][k];
			}
			//cout<<endl;
		}
		//cout<<endl;
	}

  return para;
}

double gauss1_p(double x, double para[]) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3
  double a1 = para[1];
  double m1 = para[2];
  double s1 = para[3];

  double two_pi = 2.0 * 3.1415926535;
  double sqrt2PI = sqrtf(two_pi);
  return a1*(1/(sqrt2PI*s1))*expf(-(x-m1)*(x-m1)/(2.*s1*s1));
        
}


double gauss2_p(double x, double para[]) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3
  double a1 = para[1];double a2 = para[2];
  double m1 = para[3];double m2 = para[4];
  double s1 = para[5];double s2 = para[6];

  double two_pi = 2.0 * 3.1415926535;
  double sqrt2PI = sqrtf(two_pi);
  return a1*(1/(sqrt2PI*s1))*expf(-(x-m1)*(x-m1)/(2.*s1*s1))+a2*(1/(sqrt2PI*s2))*expf(-(x-m2)*(x-m2)/(2.*s2*s2));
        
}

double gauss3_p(double x, double para[]) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3
  double a1 = para[1];double a2 = para[2];double a3 = para[3];
  double m1 = para[4];double m2 = para[5];double m3 = para[6];
  double s1 = para[7];double s2 = para[8];double s3 = para[9];

  double two_pi = 2.0 * 3.1415926535;
  double sqrt2PI = sqrtf(two_pi);
  return a1*(1/(sqrt2PI*s1))*expf(-(x-m1)*(x-m1)/(2.*s1*s1))+a2*(1/(sqrt2PI*s2))*expf(-(x-m2)*(x-m2)/(2.*s2*s2))+a3*(1/(sqrt2PI*s3))*expf(-(x-m3)*(x-m3)/(2.*s3*s3));
        
}

double gauss4_p(double x, double para[]) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3
  double a1 = para[1];double a2 = para[2];double a3 = para[3];double a4 = para[4];
  double m1 = para[5];double m2 = para[6];double m3 = para[7];double m4 = para[8];
  double s1 = para[9];double s2 = para[10];double s3 = para[11];double s4 = para[12];

  double two_pi = 2.0 * 3.1415926535;
  double sqrt2PI = sqrtf(two_pi);
  return a1*(1/(sqrt2PI*s1))*expf(-(x-m1)*(x-m1)/(2.*s1*s1))+a2*(1/(sqrt2PI*s2))*expf(-(x-m2)*(x-m2)/(2.*s2*s2))+a3*(1/(sqrt2PI*s3))*expf(-(x-m3)*(x-m3)/(2.*s3*s3));+a4*(1/(sqrt2PI*s4))*expf(-(x-m4)*(x-m4)/(2.*s4*s4));

}

double gauss5_p(double x, double para[]) {// #, a1,m1,s1,a2,m2,s2,a3,m3,s3
  double a1 = para[1];double a2 = para[2];double a3 = para[3];double a4 = para[4];double a5 = para[5];
  double m1 = para[6];double m2 = para[7];double m3 = para[8];double m4 = para[9];double m5 = para[10];
  double s1 = para[11];double s2 = para[12];double s3 = para[13];double s4 = para[14];double s5 = para[15];

  double two_pi = 2.0 * 3.1415926535;
  double sqrt2PI = sqrtf(two_pi);
  return a1*(1/(sqrt2PI*s1))*expf(-(x-m1)*(x-m1)/(2.*s1*s1))+a2*(1/(sqrt2PI*s2))*expf(-(x-m2)*(x-m2)/(2.*s2*s2))+a3*(1/(sqrt2PI*s3))*expf(-(x-m3)*(x-m3)/(2.*s3*s3));+a4*(1/(sqrt2PI*s4))*expf(-(x-m4)*(x-m4)/(2.*s4*s4))+a5*(1/(sqrt2PI*s5))*expf(-(x-m5)*(x-m5)/(2.*s5*s5));
   
}


double gen_weight_data_p(double key,  double para[]) {

    int temp = para[0];
    double Weight = 0;

    switch (temp)
    {
        case 3 :
            Weight = gauss3_p(key, para ); //longlat-around6.bin-1
            break;
        case 4 :
            Weight = gauss4_p(key, para); //longlat-around6.bin-2
            break;
        case 5 :
            Weight = gauss5_p(key, para); //gowalla-around6.bin-3
            break;
        case 2 :
            Weight = gauss2_p(key, para); //gowalla-around6.bin-3
            break;
        case 1 :
            Weight = gauss1_p(key, para); //gowalla-around6.bin-3
            break;            
        default:
            std::cout << "You did not enter A, B, or C!\n";
    }


    return Weight;
}




template <class T>
double cal_divergence(T keys[], double para1[], double para2[],int size ) {
  double divergence;
  double sum1=0;double sum2=0;

  for (int i =0; i<size; i++){
    sum1 += log(gen_weight_data_p(keys[i], para1));
    sum2 += log(gen_weight_data_p(keys[i], para2));
  }
  if(sum1> sum2){
    divergence = (sum1-sum2)/size;
  }
  else
  {
    divergence = (sum2-sum1)/size;    
  }
  

  return divergence;
}
