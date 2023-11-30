#pragma once
#include<iostream>
#include<cmath>
#include<array>
#include<string>

using namespace std;

const double pi =3.141592653589793238462643383279502884;
const double machine_eps = 2.220446049250313e-16;
const double INFTY = numeric_limits<double>::infinity();

void check_weights(double *weights,int n);
void check_x(int ns, int nc);
void check_precision_positivity_1d(double *precision,string covariance_type,int precisionx);//check cov_type
void check_precision_positivity_2d(double **precision,string covariance_type,int precisionx,int precisiony);
void check_precision_positivity_3d(double **precision,string covariance_type,int precisionx,int precisiony,int precisionz);
void check_precision_matrix(double **precision,string covariance_type,int precisionx,int precisiony);//check cov_type
void check_precision_full(double ***precision,string covariance_type,int nc,int nf);//check cov_type

void check_precision1(double *precision,string covariance_type,int nc);//check cov_type
void check_precision2d(double **precision,string covariance_type,int nc,int nf);//check cov_type
void check_precision2d(double **precision,string covariance_type,int nc,int nf);//check cov_type
void check_precision3(double ***precision,string covariance_type,int nc,int nf);//check cov_type

void estimate_gaussian_covariances_full(double** resp,double** x,double* nk,double** means,double ***covariances,double reg,int nc,int nf,int ns);//check cov_type
void estimate_gaussian_covariances_tied(double** resp,double** x,double* nk,double** means,double** covariance,double reg,int nc,int nf,int ns);//check cov_type
void estimate_gaussian_covariances_diag(double** resp,double** x,double* nk,double** means,double** covariances,double reg,int nc,int nf,int ns);//check cov_type
void estimate_gaussian_covariances_spherical(double** resp,double** x,double* nk,double** means,double* variances,double reg,int nc,int nf,int ns);//check cov_type

void estimate_gaussian_covariances_nk(double** resp,double* nk,int nc,int ns);
void estimate_gaussian_covariances_means(double** resp,double** x,double* nk,double** means,int nc,int nf,int ns);
double** estimate_gaussian_parameters_full_cov();//check cov_type//(return function for every element 3 elements ,nk,means,covariances)double*** estimate_gaussian_covariances_full+double* estimate_gaussian_covariances_nk+double** estimate_gaussian_covariances_means
double** estimate_gaussian_parameters_tied_cov();//check cov_type//(return function for every element 3 elements ,nk,means,covariances)double** estimate_gaussian_covariances_tied+double* estimate_gaussian_covariances_nk+double** estimate_gaussian_covariances_means
double** estimate_gaussian_parameters_diag_cov();//check cov_type//(return function for every element 3 elements ,nk,means,covariances)double** estimate_gaussian_covariances_diag+double* estimate_gaussian_covariances_nk+double** estimate_gaussian_covariances_means
double** estimate_gaussian_parameters_spherical_cov();//check cov_type//(return function for every element 3 elements ,nk,means,covariances)double* estimate_gaussian_covariances_spherical+double* estimate_gaussian_covariances_nk+double** estimate_gaussian_covariances_means

void compute_precision_cholesky_full(double*** covariances,double*** precision_chol,int nc,int nf);//check cov_type
void compute_precision_cholesky_tied(double** covariances,double** precision_chol,int nf);//check cov_type
void compute_precision_cholesky_diag(double** covariances,double** prec_chol,int nc,int nf);//check cov_type
void compute_precision_cholesky_spherical(double* covariances,double* prec_chol,int nc);//check cov_type

void compute_log_det_cholesky_full(double*** matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf);//check cov_type
void compute_log_det_cholesky_tied(double** matrix_chol,double log_det_chol,string covariance_type,int nf);//check cov_type
void compute_log_det_cholesky_diag(double** matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf);//check cov_type
void compute_log_det_cholesky_shperical(double* matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf);//check cov_type

void log_gaussian_prob_full(double **x,double** means,double*** precisions_chol,double** log_prob,int ns,int nc,int nf);
void log_gaussian_prob_tied(double **x,double** means,double** precisions_chol,double** log_prob,int ns,int nc,int nf);
void log_gaussian_prob_diag(double **x,double** means,double** precisions_chol,double** log_prob,int ns,int nc,int nf);
void log_gaussian_prob_spherical(double **x,double** means,double* precisions_chol,double** log_prob,int ns,int nc,int nf);

void dotArMat(double* arr1,double** arr2,double* y,int n,int m);
void dotequ(double** arr1,double** arr2,double** res,int n,int c,int m);
void linalg_cholesky(double** a,double** l,int n);
void solve_triangular(double** a,double** x,int n,int m);
void trans_matrix(double** a,double** b,int n,int m);

double mean(double* log,int n);

void summatrixarray(double** mat,double* arr,double** mat1,int ns,int nc);
void nplog(double* l,int n);
void npexp(double** l,int n,int m);
void npsum(double** l,double* sum,int n,int m);
void logsumexp(double** l,double* logsumexp,int n,int m);
void argmax(double** resp,int* max,int n_samples,int n_features);
int argmax2(double** resp,int n);
int argmin(double* resp,int n);

void check_x(int ns, int nc){
	if(nc != 0 && ns < nc){
		cerr<<"Expected n_samples >= n_components but got n_components = "<<nc <<"n_samples = "<<ns;
		exit(1);
	}
}

void check_weights(double *weights,int n){
	//check if the weights is between 0 and 1 
	// check if  1-sum of the weights>0
	//check if n is right
	double min=1,max=0,s=0;	
	for(int i=0;i<n;i++){			 
		if(min > weights[i])
			min = weights[i];
		if(max < weights[i])
			max  = weights[i];
		if(min<0 || max > 1){
			cerr<<"Error The parameter 'weights' should be in the range[0,1]  but got max value " <<max<< " min value "<<min<<" ";
		exit(1);
		}				
		s=s+weights[i];				 
	}	 

	if(abs(1-s) > exp(-8) ){
		cerr<<"Error The parameter 'weights' is not normalized ";
		exit(1);		
	 }	
}

void check_precision_positivity_1d(double *precision,string covariance_type,int precisionx){
	//check if the array is positive
	//precision x number of components

	for(int i=0;i<precisionx;i++){		 
			 if(precision[i] <= 0 ){
				 cerr<< covariance_type<<" precision should be positive, ";                      
				 exit(1);			 
		 }
	}
}
void check_precision_positivity_2d(double **precision,string covariance_type,int precisionx,int precisiony){
	//check if the array is positive
	//precision x number of components

	for(int i=0;i<precisionx;i++){
		 for(int j=0;j<precisiony;j++){
			 if(precision[i][j] <= 0 ){
				 cerr<< covariance_type<<" precision should be positive, ";
                      
				 exit(1);
			 }
			 
		 }
	}

}
void check_precision_positivity_3d(double ***precision,string covariance_type,int precisionx,int precisiony,int precisionz){
	//check if the array is positive
	//precision x number of components

	for(int i=0;i<precisionx;i++){
		 for(int j=0;j<precisionx;j++){
			 for(int k=0;k<precisionx;k++){
					if(precision[i][j][k] <= 0 ){
				 cerr<< covariance_type<<" precision should be positive, ";
                      
				 exit(1);
				}
			 }
		 }
	}

}
void check_precision_matrix(double **precision,string covariance_type,int precisionx,int precisiony){
	//check if the matrix is symmetric
	//check if it is positive_definite(eigenvalues>0)
	//missing eigenvalues functions
	// to be done later
	//precisionx : number of rows
	//precisiony : number of cols
	
	for(int i=0;i<precisionx;i++){
		 for(int j=0;j<precisiony;j++){
			 
			 if(precision[i][j] != precision[j][i]){
				
			 cerr<<covariance_type<<" precision should be symmetric, "
                         "positive-definite";
			 }
		}
	}

}

void check_precision_full(double ***precision,string covariance_type,int nc,int nf){
	//x:precisionx y:precisiony z:precisionz
	//check for every matrix in that is symmetric and positive_definite(eigenvalues>0)
	double **a = new double*[nf];
	for(int i=0;i<nf;i++)
		a[i]=new double[nf];
	for(int i=0;i<nc;i++){
		for(int j=0;j<nf;j++){
			for(int k=0;k<nf;k++){
				a[j][k] = precision[i][j][k];
		
			}
			
		}
		check_precision_matrix(a,covariance_type,nf,nf);
	}
}

void check_precision1(double *precision,string covariance_type,int nc){
	//check precision for a covariance type = spherical
	if(covariance_type.compare("spherical")){
		
		check_precision_positivity_1d(precision,covariance_type,nc);
		
	}
}
void check_precision2t(double **precision,string covariance_type,int nf){
	//check precision for a covariance type = tied
	if(covariance_type.compare("tied")){
		
		check_precision_matrix(precision,covariance_type,nf,nf);
		
	}
	
}

void check_precision2d(double **precision,string covariance_type,int nc,int nf){
		//check precision for a covariance type = diag
		
	if(covariance_type.compare("diag")){
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				if(precision[i][j] <0 ){
					cerr<< covariance_type<<" precision should be positive, ";  
					exit(1);
				}
			}
		}
	}
}


void check_precision3(double ***precision,string covariance_type,int nc,int nf){
		//check precision for a covariance type = full
	if(covariance_type.compare("full"))
		check_precision_full(precision,covariance_type,nc,nf);
	
}

void estimate_gaussian_covariances_full(double** resp,double** x,double* nk,double** means,double ***covariances,double reg,int nc,int nf,int ns){
		//resp : array, shape (n_samples, n_components)
		//X : array, shape (n_samples, n_features)
		//nk : array, shape (n_components,)
		// means : array, shape (n_components, n_features)
	/*Returns
    -------
    covariances : array, shape (n_components, n_features, n_features)
	*/
		
		double** diff=new double*[ns];
		double** arr1=new double*[nf];
		double** arr2 = new double*[nf];
		int k=0;
		double** diff1 = new double*[nf];

		for(int i=0;i<ns;i++)
			diff[i] = new double[nf];
		for(int i=0;i<nf;i++){
			arr2[i] = new double[nf];
			diff1[i] = new double[ns];
			arr1[i] = new double[ns];
		}
		while(k<nc){
			for(int i=0; i < ns;i++)
				for(int j=0;j<nf;j++)
					diff[i][j] = x[i][j] - means[k][j];
			trans_matrix(diff,diff1,ns,nf);
			
			for(int i=0; i < nf;i++){
				for(int j=0;j<ns;j++){
					arr1[i][j] = resp[j][k]*diff1[i][j];
				}
			}
			dotequ(arr1,diff,arr2,nf,ns,nf);
			for(int i=0;i<nf;i++){
				for(int j=0;j<nf;j++){
					covariances[k][i][j] =arr2[i][j]/nk[k];
					if(i==j)
					covariances[k][i][j] += reg;

				}
			}

			k++;
		}
		for(int i=0;i<ns;i++){
			delete[] diff[i];
		}
		for(int i=0;i<nf;i++){
			delete[] arr2[i];
			delete[] arr1[i];
			delete[] diff1[i];
		}
		delete[] diff1;
		delete[] arr2;
		delete[] arr1;
		delete[] diff;

	}
void estimate_gaussian_covariances_tied(double** resp,double** x,double* nk,double** means,double** covariance,double reg,int nc,int nf,int ns){
		//n : number of rows in x
		//m : number of cols in x 
		//meansr : number of rows in means
		//meansc : number of cols in means
		//nkx : number of components in nk
		
		//resp : array, shape (n_samples, n_components)
		// X : array, shape (n_samples, n_features)
	    // nk : array, shape (n_components,)
	    // means : array, shape (n_components, n_features)
		/*Returns
		-----------
		 covariance : array, shape (n_features, n_features)
		*/
		double **meanst = new double*[nf];
		double **avg_means = new double*[nf];
		double s=0;
		
		double **xt = new double*[nf];
		double **avg_x2 = new double*[nf];
	
		
		for(int i=0;i<nf;i++){
			meanst[i] = new double[nc];
			avg_means[i] = new double[nf];
			xt[i] = new double[ns];
			avg_x2[i] = new double[nf];
		
		}
		trans_matrix(x,xt,ns,nf);
		dotequ(xt,x,avg_x2,nf,ns,nf);
		trans_matrix(means,meanst,nc,nf);
		for(int i=0;i<nf;i++){
			for(int j=0;j<nc;j++){
				meanst[i][j] = meanst[i][j] *nk[j];
			}
		}
		dotequ(meanst,means,avg_means,nf,nc,nf);
		for(int i=0;i<nf;i++){
			for(int j=0;j<nf;j++){
				covariance[i][j] = avg_x2[i][j] - avg_means[i][j];


			}
		}
		for(int i=0;i<nc;i++){
			s=s+nk[i];
		}
		for(int i=0;i<nf;i++){
			for(int j=0;j<nf;j++){
				covariance[i][j] = covariance[i][j]/s;
				if(i==j)
				covariance[i][j] += reg;
			}
		}
		for(int i=0;i<nf;i++){
			delete[] meanst[i];
			delete[] avg_means[i];
			delete[] xt[i];
			delete[] avg_x2[i];
		}
		delete[] meanst;
		delete[] avg_means;
		delete[] xt;
		delete[] avg_x2;
		


}
void estimate_gaussian_covariances_diag(double** resp,double** x,double* nk,double** means,double** covariances,double reg,int nc,int nf,int ns){
		

		//responsibilities : array, shape (n_samples, n_components)
		 // X : array, shape (n_samples, n_features)
		 //nk : array, shape (n_components,)
		// means : array, shape (n_components, n_features)

		//Returns
		// -------
		//covariances : array, shape (n_components, n_features)
		double** avg_means2 = new double*[nc];
		double** x2 = new double*[ns];
		double** avg_x2 = new double*[nc];
		double** avg_x_means = new double*[nc];
		
		
		
		for(int i=0;i<ns;i++)
			x2[i] = new double[nf];

		
		for(int i=0;i<nc;i++){
			avg_means2[i] = new double[nf];
			avg_x2[i] = new double[nf];
			avg_x_means[i] = new double[nf];
			
		}

		for(int i=0;i<ns;i++){
			for(int j=0;j<nf;j++){
				x2[i][j] = x[i][j]*x[i][j];
				
			}
		}
		trans_matrix(resp,resp,ns,nc);
		dotequ(resp,x2,avg_x2,nc,ns,nf);

		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				avg_x2[i][j] = avg_x2[i][j] / nk[i];
			}
		}
		

		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				avg_means2[i][j] = means[i][j] * means[i][j];
			}
		}
		dotequ(resp,x,avg_x_means,nc,ns,nf);
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				avg_x_means[i][j] = means[i][j] * avg_x_means[i][j] / nk[i]; 
				covariances[i][j] =avg_x2[i][j] - 2*avg_x_means[i][j] + avg_means2[i][j] + reg;
			}
		}
		for(int i=0;i<ns;i++){
			delete[] x2[i];
		}
		for(int i=0;i<nc;i++){
			delete[] avg_means2[i];
			delete[] avg_x2[i];
			delete[] avg_x_means[i];
			
		}
	delete[] x2;
	delete[] avg_means2;
	delete[] avg_x2;
	delete[] avg_x_means;		
	}
void estimate_gaussian_covariances_spherical(double** resp,double** x,double* nk,double** means,double* variances,double reg,int nc,int nf,int ns){
		//n: rows in x
		//m: cols in x
		//int respn = rows in resp
		//int respm = cols in resp
		//meansr = rows in means
		//meansc = cols in means
	/*    Parameters
    ----------
    responsibilities : array, shape (n_samples, n_components)

    X : array, shape (n_samples, n_features)

    nk : array, shape (n_components,)

    means : array, shape (n_components, n_features)

    Returns
    -------
    variances : array, shape (n_components,)
	*/
		double s=0;
		double** arr = new double*[nc];
		for(int i=0;i<nc;i++)
			arr[i] = new double[nf];
		
		estimate_gaussian_covariances_diag(resp,x,nk,means,arr,reg,nc,nf,ns);//return shape(n_components,n_features)
		
		for(int i=0;i<nc;i++){
			s=0;
			for(int j=0;j<nf;j++){
				s=s+arr[i][j];
			}
			variances[i] = s/nf;
		}
		for(int i=0;i<nc;i++)
			delete[] arr[i];

		delete[] arr;
	}
void estimate_gaussian_covariances_nk(double** resp,double* nk,int nc,int ns){
	
	//resp : array, shape (n_samples, n_components)

	//Returns 
	//------
	// nk : array, shape (n_components)
	double s=0;

	for(int i=0;i<nc;i++){
		s=0;
		for(int j=0;j<ns;j++){
			s += resp[j][i] ;
		}
		nk[i] = s + 10 *machine_eps;
	}
}
void estimate_gaussian_covariances_means(double** resp,double** x,double* nk,double** means,int nc,int nf,int ns){
	//respr : rows in resp
	//respc : columns in resp
	//xr    : rows in x
	//xc    : columns in x
	//nk_nc : number of components in nk

	/*
	 X : array, shape (n_samples, n_features)
        The input data array.
    resp : array, shape (n_samples, n_components)
        The responsibilities for each data sample in X.

	 Returns
    -------
    means : array, shape (n_components, n_features)*/
	double** respt = new double*[nc];
	double** c = new double* [nc];

	for(int i=0;i<nc;i++){
		respt[i] = new double[ns];
		c[i] = new double[nf];
	}

	trans_matrix(resp,respt,ns,nc);
	dotequ(respt,x,c,nc,ns,nf);
	for(int i =0;i<nc;i++){
		for(int j=0;j<nf;j++){
			means[i][j] = c[i][j]/nk[i];
		}
	}
	for(int i=0;i<nc;i++){
		delete[] respt[i];
		delete[] c[i];
	}
	delete[] respt;
	delete[] c;
}
void compute_precision_cholesky_full(double*** covariances,double*** precision_chol,int nc,int nf){
		

		//int n = rows in covariances
		//int m = cols in covariances

		//Return precision_chol shape(nc,nf,nf)
	
		double** cov_chol = new double*[nf];
		double** cov_chol1 = new double*[nf];
		double** a = new double*[nf];
		double** d = new double*[nf];
		for(int i=0;i<nf;i++){
			cov_chol[i] = new double[nf];
			cov_chol1[i] = new double[nf];
			a[i] = new double[nf];
			d[i] = new double[nf];

		}
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				for(int k=0;k<nf;k++){
					cov_chol[j][k] = covariances[i][j][k];
				}
			}

			
			linalg_cholesky(cov_chol,cov_chol1,nf);
			solve_triangular(cov_chol1,a,nf,nf);
			trans_matrix(a,d ,nf,nf);
			for(int j=0;j<nf;j++)
				for(int k=0;k<nf;k++)
					precision_chol[i][j][k] = d[j][k];
		}
		for(int i=0;i<nf;i++){
			delete[] a[i];
			delete[] cov_chol[i];
			delete[] cov_chol1[i];
			delete[] d[i];
		}
		delete[] d;
		delete[] a;
		delete[] cov_chol;
		delete[] cov_chol1;
	}
void compute_precision_cholesky_tied(double** covariances,double** precision_chol,int nf){
		

		//covariances array,shpace(nc,nf)
		
		double** cov_chol = new double*[nf];
		
		for(int i=0;i<nf;i++){
			cov_chol[i] = new double[nf];
			
		}
		linalg_cholesky(cov_chol,cov_chol,nf);
			
		solve_triangular(cov_chol,cov_chol,nf,nf);
		trans_matrix(cov_chol,precision_chol,nf,nf);
		for(int i=0;i<nf;i++){
			
			delete[] cov_chol[i];
		}
		
		delete[] cov_chol;

	}

void compute_precision_cholesky_diag(double** covariances,double** prec_chol,int nc,int nf){
		//we should make try and exception(raise value error )
		string estimate_precision_error_message = 
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar.";
		//covariances array,shpace(nc,nf)
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				if(covariances[i][j] <=0 ){
					cerr<<estimate_precision_error_message;
					exit(1);
				}
				prec_chol[i][j] = 1 / sqrt(covariances[i][j]) ;

			}
		}
}
		
void compute_precision_cholesky_spherical(double* covariances,double* prec_chol,int nc){
	//we should make try and exception(raise value error )
	string estimate_precision_error_message = 
    "Fitting the mixture model failed because some components have "
    "ill-defined empirical covariance (for instance caused by singleton "
    "or collapsed samples). Try to decrease the number of components, "
    "or increase reg_covar.";
	//int n = rows in covariances
	//int m = cols in covariances
	
	for(int i=0;i<nc;i++){
		if(covariances[i] <=0 ){
			cerr<<estimate_precision_error_message;
			exit(1);
		}
		prec_chol[i] = 1 / sqrt(covariances[i]) ;
	}
	
}
void compute_log_det_cholesky_full(double*** matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf){
		// matrix_chol : array,shape of (n_components, n_features, n_features)
		
		double** a =new double*[nc];
		int z=0;

		for(int i=0;i<nc;i++){
			log_det_chol[i] =0;
			a[i] = new double[nf*nf];
		}
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				for(int k=0;k<nf;k++){
					a[i][j*nf +k] = matrix_chol[i][j][k];
					
				}
			}
			
		}
			for(int i=0;i<nc;i++){
				for(int j=0;j<nf*nf;j=j+nf+1){
					a[i][j] = log(a[i][j]);
					log_det_chol[i] = log_det_chol[i] + a[i][j];
				}
		}
			for(int i=0;i<nc;i++){
				delete[] a[i];
			}
			delete[] a;
		
}
void compute_log_det_cholesky_tied(double** matrix_chol,double log_det_chol,string covariance_type,int nf){
		// matrix_chol : array,shape of (n_features, n_features)
		log_det_chol =0 ;
		for(int i=0;i<nf;i++){
			log_det_chol =log_det_chol + log(matrix_chol[i][i]);
	
		}
		

	}
void compute_log_det_cholesky_diag(double** matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf){
		//nc number of rows and nf number of columns
		
		for(int i=0;i<nc;i++){
			for(int j=0;j<nf;j++){
				log_det_chol[i] = 0;
				log_det_chol[i] += log(matrix_chol[i][j]);
			}

		}
		
		
	}
void compute_log_det_cholesky_shperical(double* matrix_chol,double* log_det_chol,string covariance_type,int nc,int nf){
		// matrix_chol : array,shape of (n_components, n_features,)

		
		for(int i=0;i<nc;i++)
			log_det_chol[i] = matrix_chol[i] * nf;

		
	}
void log_gaussian_prob_full(double **x,double** means,double*** precisions_chol,double** log_prob,int ns,int nc,int nf){
		// ns = number of samples
		// nc = number of componenents
		// nf = number of features
		//x array, shape (ns,nf)
		//means array, shape  (nc,nf)
		//precisions_chol array,shape (nc,nf,nf)
		//Returns
		//-----
		//log_prob array,shape (ns,nc)
		double* log_det = new double[nc];
		compute_log_det_cholesky_full(precisions_chol,log_det,"full",nc,nf);//log_det shape = (nc)
		
		double** y = new double*[ns];
		double** x1 = new double*[ns];
		double *x2 = new double[ns];
		double* y1 = new double[nf];
		double* mu = new double[nf];
		double* y3 = new double[nf];
		for(int i=0;i<ns;i++){
			y[i] = new double[nf];
			x1[i] = new double[nf];
		}
		for(int i=0;i<nc;i++){
			dotequ(x,precisions_chol[i],y,ns,nf,nf);
			for(int j=0;j<nf;j++)
				mu[j] = means[i][j];
			dotArMat(mu,precisions_chol[i],y1,nf,nf);
			for(int k=0;k<ns;k++){
				x2[k]=0;
				for(int j=0;j<nf;j++){
					x1[k][j] = y[k][j] -y1[j];
					x1[k][j] = x1[k][j]*x1[k][j];
					x2[k] =x2[k] + x1[k][j];
				}
			}
			for(int j=0;j<ns;j++)
				log_prob[j][i] = x2[j];
		}
		for(int i=0;i<ns;i++)
			for(int j=0;j<nc;j++)
				log_prob[i][j] = -0.5*(nf*log(2*pi)+log_prob[i][j]) + log_det[j];
		for(int i=0;i<ns;i++){
			delete[] y[i];
			delete[] x1[i];
		}
		delete[] y;
		delete[] x1;
		delete[] y1;
		delete[] x2;
		delete[] y3;
		delete[] mu;
		delete[] log_det;
	}
void log_gaussian_prob_tied(double **x,double** means,double** precisions_chol,double** log_prob,int ns,int nc,int nf){
		// ns = number of samples
		// nc = number of componenents
		// nf = number of features
		//x array, shape (ns,nf)
		//means array ,shape  (nc,nf)
		//precisions_chol array,shape  (nf,nf)
		//Returns 
		//------
		//log_prob array,shape(ns,nc)
		double** y =new double*[ns];;
		
		double* y1 = new double[nf];;
		double* mu = new double[nf];
		double* x2 =new double[ns];
		double log_det =0; 
		compute_log_det_cholesky_tied(precisions_chol,log_det,"tied",nf);
		for(int i=0;i<ns;i++)
			y[i] = new double[nf];
		dotequ(x,precisions_chol,y,ns,nf,nf);
		for(int i=0;i<nc;i++){
			
			for(int j=0;j<nf;j++){
				mu[j] = means[i][j];
				}
			dotArMat(mu,precisions_chol,y1,nf,nf);
			for(int j=0;j<ns;j++){
				x2[j]=0;
				for(int k=0;k<nf;k++){
					y[j][k] = y[j][k] -y1[k];
					y[j][k] = y[j][k]*y[j][k];
					x2[j] = x2[j] +y[j][k];
					
					}
				log_prob[j][i] =x2[j];

				}
			}
		for(int i=0;i<ns;i++){
			for(int j=0;j<nc;j++){
				log_prob[i][j] = -0.5*(nf*log(2*pi)+log_prob[i][j]) + log_det;
			}
		}
		for(int i=0;i<ns;i++)
			delete[] y[i];
		delete[] y;
		delete[] y1;
		delete[] mu;
		delete[] x2;

		}
void log_gaussian_prob_diag(double **x,double** means,double** precisions_chol,double** log_prob,int ns,int nc,int nf){
		// ns = number of samples
		// nc = number of componenents
		// nf = number of features
		//x array,shape (ns,nf)
		//means array, shape  (nc,nf)
		//precisions_chol array, shape (nc,nf)
		//returns
		//------
		//log_prob array,shape(ns,nc)
		double* log_det = new double[nc];; 
		compute_log_det_cholesky_diag(precisions_chol,log_det,"diag",nc,nf);
		double** precisions = new double*[nc];
		double** mu = new double*[nc];
		double* s = new double[nc];
		double** mt = new double*[nc] ;
		double** mt1= new double*[nf];
		double** mt2= new double*[ns];
		double** xp = new double*[ns];
		double** x2 = new double*[ns];
		
		double** tp =new double*[nf];
		for(int i=0;i<ns;i++){
			x2[i] = new double[nf];
			mt2[i] = new double [nc];
			xp[i] = new double[nc];
		}
		for(int i=0;i<nf;i++){
			tp[i] = new double[nc];
			mt1[i] = new double[nc];
		}
		for(int i=0;i<nc;i++){
			precisions[i] = new double[nf];
			mu[i] = new double[nf];
			mt[i] = new double[nf];
			
		}
		// s = np.sum((means ** 2 * precisions), 1)
		for(int i=0;i<nc;i++){
			s[i]=0;
			for(int j=0;j<nf;j++){
				precisions[i][j] = precisions_chol[i][j] *precisions_chol[i][j];
				mu[i][j] = means[i][j] * means[i][j] * precisions[i][j];
				s[i]=s[i]+mu[i][j];
				//mt2 = 2. * np.dot(X, (means * precisions).T)
				mt[i][j] =means[i][j] *precisions[i][j];
			}
			
		}
		trans_matrix(mt,mt1,nc,nf);
		
		dotequ(x,mt1,mt2,ns,nf,nc);
		for(int i=0;i<ns;i++){
			for(int j=0;j<nc;j++){
				mt2[i][j] = 2 * mt2[i][j];
			}
			
		}
		// xp = np.dot(X ** 2, precisions.T)

		trans_matrix(precisions,tp,nc,nf);
		for(int i=0;i<ns;i++){
			for(int j=0;j<nf;j++){
				x2[i][j] = x[i][j] *x[i][j];
			}
		}
		dotequ(x2,tp,xp,ns,nf,nc);
		
		for(int i=0;i<ns;i++){
			for(int j=0;j<nc;j++){
				log_prob[i][j] = s[j] -mt2[i][j] + xp[i][j];
				log_prob[i][j] = -0.5*(nf*log(2*pi)+log_prob[i][j]) + log_det[j];
			}
		}
		for(int i=0;i<ns;i++){
			delete[] x2[i];
			delete[] mt2[i];
			delete[] xp[i];
		}
		for(int i=0;i<nf;i++){
			delete[] tp[i];
			delete[] mt1[i];
		}
		for(int i=0;i<nc;i++){
			delete[] precisions[i];
			delete[] mu[i];
			delete[] mt[i];
			
		}
		delete[] s;
		delete[] x2;
		delete[] mt2;
		delete[] xp;
		delete[] tp;
		delete[] mt1;
		delete[] precisions;
		delete[] mu;
		delete[] mt;
		delete[] log_det;

	}
void log_gaussian_prob_spherical(double **x,double** means,double* precisions_chol,double** log_prob,int ns,int nc,int nf){
		// ns = number of samples
		// nc = number of componenents
		// nf = number of features
		//x array,shape (ns,nf)
		//means array,shape  (nc,nf)
		//precisions_chol array,  (nc)
		//Returns 
		//-------
		//log_prob array,shape (ns,nc)
		double* precision = new double[nc];
		double** means2 = new double*[nc];

		double** npo= new double*[ns];
		double* sm = new double[nc];
		double** mt=new double *[nf];
		double** mt2 =new double*[ns];
		double** x2 = new double*[ns];
		double* x3 = new double[ns];
		double* log_det = new double[nc];
		compute_log_det_cholesky_shperical(precisions_chol,log_det,"spherical",nc,nf);
		//sm = np.sum(means ** 2, 1) * precisions
		for(int i=0;i<nf;i++)
			mt[i] = new double[nc];
		for(int i=0;i<nc;i++){
			mt2[i] = new double[nc];
			sm[i]=0;
			means2[i] = new double[nf];
			precision[i] = precisions_chol[i] * precisions_chol[i];
			for(int j=0;j<nf;j++){
				means2[i][j] = means[i][j] * means[i][j];
				sm[i] = sm[i] + means[i][j];
			}
			
		}
		// mt2 = 2 * np.dot(X, means.T * precisions)
		trans_matrix(means,mt,nc,nf);
		for(int i=0;i<nf;i++){
			for(int j=0;j<nc;j++){
				mt[i][j] = mt[i][j] * precision[j];
			}
		}
		dotequ(x,mt,mt2,ns,nf,nc);
		// npo = np.outer( x3 = (row_norms(X, squared=True)), precisions)
		for(int i=0;i<ns;i++){
			for(int j=0;j<nc;j++){
				mt2[i][j] = 2 * mt2[i][j];
			}
		}
		for(int i=0;i<ns;i++)
			x2[i] = new double[nf];
		for(int i=0;i<ns;i++){
			x3[i]=0;
			for(int j=0;j<nf;j++){
				x2[i][j] =x[i][j] * x[i][j];
				x3[i] = x3[i] + x2[i][j];
			}
		}
		for(int i=0;i<ns;i++){
			npo[i] = new double[nc];
		
			for(int j=0;j<nc;j++){
				npo[i][j] = x3[i] * precision[j];
			}
		}
		for(int i=0;i<ns;i++){
			for(int j=0;j<nc;j++){
				log_prob[i][j] = sm[j] - mt2[i][j] + npo[i][j];
				log_prob[i][j] = -0.5*(nf*log(2*pi)+log_prob[i][j]) + log_det[j];

			}
		}
	
		for(int i=0;i<nf;i++)
			delete[] mt[i];
		for(int i=0;i<nc;i++)
			delete[] means2[i];
		for(int i=0;i<ns;i++){
			delete[] mt2[i];
			delete[] x2[i];
			delete[] npo[i];
		}

		delete[] means2;
		delete[] mt;
		delete[] sm;
		delete[] precision;
		delete[] x3;
		delete[] log_det;
	}

void dotArMat(double* arr1,double** arr2,double* y,int n,int m){
	//n : number of cols in array2
	//m : number of rows in array2
	//Return y shape(m)
	for(int i=0;i<m;i++){
		y[i] =0;
		for(int j=0;j<n;j++){
			y[i] = y[i] +arr1[j] * arr2[j][i];
		}
	}
}

void dotequ(double** arr1,double** arr2,double** res,int n,int c,int m){
	//arr1(n,c).arr2(c,m) = res(n,m)
	//matrix multiplication function arr1.arr2 = res
	// n : number of rows in arr1
	// m : number of cols in arr2 
	// c: number of rows in arr2 = cols in arr1
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			res[i][j] = 0;
			for(int k=0;k<c;k++){
				res[i][j] +=arr1[i][k] * arr2[k][j];
			}
		}
	}		
}
	


void linalg_cholesky(double** a,double** l,int n){
		//cholesky decomposition calculation of a matrix
		// a should be positive-definite(eigenvalues >0)
		// return l / a = l.l* / l lower triangle matrix
		// n: number of rows in a 
	double s=0,s1=0;
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			l[i][j] =0;	
	
	for(int i=0;i<n;i++){
		for(int j=0;j<=i;j++){
		s=0;
		s1=0;
		if(j==i){
			for(int p=0;p<j;p++)
				s = s+l[j][p]*l[j][p];
			l[i][i] = sqrt(a[i][i]-s);

			}
		else{
			for(int p=0;p<j;p++)
				s1 =s1 +l[i][p]*l[j][p];

			l[i][j] =(a[i][j] - s1)/l[j][j];

			}
		}
		
	}
}

void solve_triangular(double** a,double** x,int n,int m){
	/*1. for k = 1 to n
2.   X[k,k] = l/L[k,k]
3.   for i = k+1 to n
4.     X[i,k] = -L[i, k:i-1]*X[k:i-1,k]/L[i,i]
5.   end for i
6. end for k */
	//solve ax = b where a = lower triangular matrix and b = ones on the diagonal else zeros inverse of a

	
	
	double s;
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			x[i][j] =0;
		}
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<m;j++){
			s=0;
			if(i==j)
				x[i][i] = 1/a[i][i];
			else
				if(i<j)
					x[i][j] =0;
				else{
					for(int k=j;k<i;k++){
						s += a[i][k] * x[k][j] ;

					}
					x[i][j] = (-1*s)/a[i][i];
					
				}
				
		}
	}
	
}

void trans_matrix(double** a,double** b,int n, int m){
	//n : number of rows 
	//m: number of columns
	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++)
			b[j][i] = a[i][j];
}

double mean(double* log,int n){
	double avg = 0;
	for (int i = 0; i < n; i++)
		avg+=log[i];
	avg/=n;
	return avg;
}

void summatrixarray(double** mat,double* arr,double** mat1,int ns,int nc){
	//return mat1 shape(ns,nc)
	for (int i=0; i<ns; i++) 
      for (int j=0; j<nc; j++) 
         mat1[i][j] =mat[i][j] + arr[j];
	
}

void nplog(double* l,double * logl,int n){
	for (int i = 0; i < n; i++)
		logl[i]=log(l[i]);
}
void npexp(double** l,double** expl,int n,int m){
	for (int i = 0; i < n; i++)
		for (int j = 0; j < m; j++)
			expl[i][j]=exp(l[i][j]);
	
}
void npsum(double** l,double* sum,int n,int m){
	for (int i = 0; i < n; i++){
		sum[i]=0;
		for (int j = 0; j < m; j++)
			sum[i]+=l[i][j];
	}
}
void logsumexp(double** l,double* logsumexp,int n,int m){
	double** expl = new double*[n];
	for (int i = 0; i < n; i++)
		expl[i] = new double[m];
	npexp(l,expl,n,m);
	npsum(expl, logsumexp,n,m);
	for (int i = 0; i < n; i++)
		logsumexp[i]=log(logsumexp[i]);
	for (int i = 0; i < n; i++)
		delete[] expl[i];
	delete[] expl;
}
void argmax(double** resp,int* max,int n_samples,int n_features){
	for (int i = 0; i < n_samples; i++){
		max[i]=0;
		for (int j = 1; j < n_features; j++)
			if (resp[i][j] > resp[i][max[i]])
				max[i]=j;
	}
}
int argmin(double* resp,int n){
	int min=0;
	for (int i = 1; i < n; i++)
		if (resp[i] < resp[min])
			min=i;
	return min;
}
int argmax2(double* resp,int n){
	int min=0;
	for (int i = 0; i < n; i++)
		if (resp[i] > resp[min])
			min=i;
	return min;
}

