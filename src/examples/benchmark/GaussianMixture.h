#include<iostream>
#include<cmath>
#include<array>
#include<string>
#include<limits>
#include<time.h>
#include "GMFunctions.h"
#include "KMeans.h"
using namespace std;

class GaussianMixture{
	/*
	Gaussian Mixture.

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution.
	*/
private:
	/*
	Attributes
    ----------
    weights_ : double[n_components]
        The weights of each mixture components.

    means_ : double[n_components][n_features]
        The mean of each mixture component.

    covariances_ : double[][][]
        The covariance of each mixture component.
        The shape depends on `covariance_type`::

            [n_components]							if 'spherical',
			[n_features][n_features]				if 'tied',
			[n_components][n_features]				if 'diag',
			[n_components][n_features][n_features]	if 'full'

    precisions_ : double[][][]
        The precision matrices for each component in the mixture. A precision
        matrix is the inverse of a covariance matrix. A covariance matrix is
        symmetric positive definite so the mixture of Gaussian can be
        equivalently parameterized by the precision matrices. Storing the
        precision matrices instead of the covariance matrices makes it more
        efficient to compute the log-likelihood of new samples at test time.
        The shape depends on `covariance_type`::

            [n_components]							if 'spherical',
			[n_features][n_features]				if 'tied',
			[n_components][n_features]				if 'diag',
			[n_components][n_features][n_features]	if 'full'

    precisions_cholesky_ : double[][][]
        The cholesky decomposition of the precision matrices of each mixture
        component. A precision matrix is the inverse of a covariance matrix.
        A covariance matrix is symmetric positive definite so the mixture of
        Gaussian can be equivalently parameterized by the precision matrices.
        Storing the precision matrices instead of the covariance matrices makes
        it more efficient to compute the log-likelihood of new samples at test
        time. The shape depends on `covariance_type`::

            [n_components]							if 'spherical',
			[n_features][n_features]				if 'tied',
			[n_components][n_features]				if 'diag',
			[n_components][n_features][n_features]	if 'full'

    converged_ : bool
        True when convergence was reached in train(), False otherwise.

    n_iter_ : int
        Number of step used by the best train of EM to reach the convergence.

    lower_bound_ : double
        Lower bound value on the log-likelihood (of the training data with
        respect to the model) of the best train of EM.
	*/
	double 
		**X,

		*weights_,
		**means_,

		*covariances_spherical_,
		**covariances_tied_,
		**covariances_diag_,
		***covariances_full_,

		*precisions_spherical_,
		**precisions_tied_,
		**precisions_diag_,
		***precisions_full_,

		*precisions_cholesky_spherical_,
		**precisions_cholesky_tied_,
		**precisions_cholesky_diag_,
		***precisions_cholesky_full_,

		lower_bound_;

	int n_iter_;

	bool converged_;
	
	string 
		covariance_type,
		init_params;

	unsigned int 
		rndseed;

	int 
		n_components,
		n_samples,
		n_features,
		max_iter,
		n_init,
		verbose,
		verbose_interval;
	double 
		tol,
		reg_covar,
		*weights_init,
		**means_init,

		*precisions_init_spherical,
		**precisions_init_tied,
		**precisions_init_diag,
		***precisions_init_full;

	bool warm_start;

	time_t init_prev_time,iter_prev_time;

	void train_predict(double **X  ,int* labels);
	void check_initial_parameters();
	void check_parameters();
	void initialize_parameters(double **X);
	void initialize(double **X,double **resp);
	void e_step(double **X,double &log_prob_norm,double**log_responsibility);
	void estimate_log_prob_resp(double **X,double* log_prob_norm,double**log_responsibility);
	void estimate_weighted_log_prob(double **X,double** a);
	void estimate_log_prob(double **X,double** log_prob);
	void estimate_log_weights(double *we);
	void m_step(double**X,double** log_resp);
	double compute_lower_bound(double** _,double log_prob_norm);
	void print_verbose_msg_iter_end(int n_iter,double diff_ll);
	void print_verbose_msg_init_end(double ll);
    void print_verbose_msg_init_beg(int n_init);
	void get_parameters_spherical_(double*&weights,double**&means,double*&covariances);
	void get_parameters_tied_(double*&weights,double**&means,double**&covariances);
	void get_parameters_diag_(double*&weights,double**&means,double**&covariances);
	//void get_parameters_full_(double*&weights,double**&means,double***&covariances);
	void set_parameters_spherical_(double*&weights,double**&means,double*&covariances,double*&precisions_cholesky,int n_features);
	void set_parameters_tied_(double*&weights,double**&means,double**&covariances,double**&precisions_cholesky,int n_features);
	void set_parameters_diag_(double*&weights,double**&means,double**&covariances,double**&precisions_cholesky,int n_features);
	void set_parameters_full_(double*&weights,double**&means,double***&covariances,double***&precisions_cholesky,int n_features);
	void set_parameters();
	int n_parameters();
	void score_samples(double* s);
public:
	/*
		Parameters
		----------
		n_components : int, defaults to 1.
			The number of mixture components.
		covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}
			String describing the type of covariance parameters to use.
			Must be one of:

			'full'
			    each component has its own general covariance matrix
			'tied'
			    all components share the same general covariance matrix
			'diag'
			    each component has its own diagonal covariance matrix
			'spherical'
			    each component has its own single variance

		tol : double, defaults to 1e-3.
			The convergence threshold. EM iterations will stop when the
			lower bound average gain is below this threshold.

		reg_covar : double, defaults to 1e-6.
			Non-negative regularization added to the diagonal of covariance.
			Allows to assure that the covariance matrices are all positive.

		max_iter : int, defaults to 100.
			The number of EM iterations to perform.

		n_init : int, defaults to 1.
			The number of initializations to perform. The best results are kept.

		init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
			The method used to initialize the weights, the means and the
			precisions.
			Must be one of::
				
				'kmeans' : responsibilities are initialized using kmeans.
				'random' : responsibilities are initialized randomly.

		weights_init : double[n_components] , optional
		    The user-provided initial weights, defaults to None.
		    If it None, weights are initialized using the `init_params` method.

		means_init : double[n_components][n_features] , optional
			The user-provided initial means, defaults to None,
			If it None, means are initialized using the `init_params` method.

		precisions_init : double[n_components][n_features][n_features], optional.
		    The user-provided initial precisions (inverse of the covariance
		    matrices), defaults to None.
		    If it None, precisions are initialized using the 'init_params' method.
		    The shape depends on 'covariance_type'::
				
				[n_components]							if 'spherical',
				[n_features][n_features]				if 'tied',
				[n_components][n_features]				if 'diag',
				[n_components][n_features][n_features]	if 'full'

		rndseed : int, random starting index or 0, optional (default=0)

		warm_start : bool, default to False.
		    If 'warm_start' is True, the solution of the last training is used as
		    initialization for the next call of train(). This can speed up
		    convergence when train is called several times on similar problems.
		    In that case, 'n_init' is ignored and only a single initialization
		    occurs upon the first call.
		
		verbose : int, default to 0.
		    Enable verbose output. If 1 then it prdoubles the current
		    initialization and each iteration step. If greater than 1 then
		    it prdoubles also the log probability and the time needed
		    for each step.

		verbose_interval : int, default to 10.
			Number of iteration done before the next prdouble.

		*/
	GaussianMixture(int n_components=1, int n_samples=1, int n_features=1, string covariance_type="full", 
	int max_iter=100, double tol=1e-3, double reg_covar=1e-6, unsigned int rndseed=0, int verbose=0,
	int verbose_interval=10, int n_init=1, string init_params="kmeans");

	~GaussianMixture();

	void train(double **X, int* labels, int n_samples, int n_features);
	void predict_proba(double **X, double** probs, int n_samples, int n_features);
	double bic();
	double aic();
	double log_likelihood();
	void get_parameters_full_(double*&weights,double**&means,double***&covariances);
};

GaussianMixture::GaussianMixture(int n_components, int n_samples, int n_features, string covariance_type, 
	int max_iter, double tol, double reg_covar, unsigned int rndseed, int verbose, int verbose_interval,
	int n_init, string init_params){

	this->n_components=n_components;
	this->n_samples=n_samples;
	this->n_features=n_features;
	this->covariance_type = covariance_type;
	this->max_iter=max_iter;
	this->tol=tol;
	this->reg_covar=reg_covar;
	this->rndseed=rndseed;
	this->verbose=verbose;
	this->verbose_interval=verbose_interval;
	this->n_init=n_init;
	this->init_params=init_params;
    this->weights_init = NULL;
    this->means_init = NULL;
	this->precisions_init_full=NULL;
	this->precisions_init_diag=NULL;
	this->precisions_init_tied=NULL;
	this->precisions_init_spherical=NULL;
	this->warm_start=false;

	this->X = NULL;
	this->weights_ = NULL;
	this->means_ = NULL;
	this->covariances_spherical_ = NULL;
	this->covariances_tied_ = NULL;
	this->covariances_diag_ = NULL;
	this->covariances_full_ = NULL;
	this->precisions_cholesky_spherical_ = NULL;
	this->precisions_cholesky_tied_ = NULL;
	this->precisions_cholesky_diag_ = NULL;
	this->precisions_cholesky_full_ = NULL;
	this->precisions_full_ = NULL;
	this->precisions_tied_ = NULL;
	this->precisions_spherical_ = NULL;
	this->precisions_diag_ = NULL;
}

GaussianMixture::~GaussianMixture(){
	for(int i=0; i<n_components; i++){
		if(means_!=NULL)
			delete means_[i];
		if(covariances_diag_ != NULL)
			delete covariances_diag_[i];
		for(int j=0; j<n_features; j++){
			if(covariances_full_ != NULL)
				delete covariances_full_[i][j];
			if(precisions_cholesky_full_ != NULL)
				delete precisions_cholesky_full_[i][j];
			if(precisions_full_ != NULL)
				delete precisions_full_[i][j];
		}
		if(covariances_full_ != NULL)
			delete covariances_full_[i];
		if(precisions_cholesky_diag_ != NULL)
			delete precisions_cholesky_diag_[i];
		if(precisions_cholesky_full_ != NULL)
			delete precisions_cholesky_full_[i];
		if(precisions_full_ != NULL)
			delete precisions_full_[i];
		if(precisions_diag_ != NULL)
			delete precisions_diag_[i];
	}
	for(int i=0; i<n_features; i++){
		if(covariances_tied_ != NULL)
			delete covariances_tied_[i];
		if(precisions_cholesky_tied_ != NULL)
			delete precisions_cholesky_tied_[i];
		if(precisions_tied_ != NULL)
			delete precisions_tied_[i];
	}
	if(weights_ != NULL)
		delete weights_;
	if(means_!=NULL)
		delete means_;
	if(covariances_spherical_ != NULL)
		delete covariances_spherical_;
	if(covariances_tied_ != NULL)
		delete covariances_tied_;
	if(covariances_diag_ != NULL)
		delete covariances_diag_;
	if(covariances_full_ != NULL)
		delete covariances_full_;
	if(precisions_cholesky_spherical_ != NULL)
		delete precisions_cholesky_spherical_;
	if(precisions_cholesky_tied_ != NULL)
		delete precisions_cholesky_tied_;
	if(precisions_cholesky_diag_ != NULL)
		delete precisions_cholesky_diag_;
	if(precisions_cholesky_full_ != NULL)
		delete precisions_cholesky_full_;
	if(precisions_full_ != NULL)
		delete precisions_full_;
	if(precisions_tied_ != NULL)
		delete precisions_tied_;
	if(precisions_spherical_ != NULL)
		delete precisions_spherical_;
	if(precisions_diag_ != NULL)
		delete precisions_diag_;
}
	
void GaussianMixture::train(double **X, int* labels, int n_samples, int n_features){
	/*
	Estimate model parameters with the EM algorithm.

    The method trains the model ``n_init`` times and sets the parameters with
    which the model has the largest likelihood or lower bound. Within each
    trial, the method iterates between E-step and M-step for ``max_iter``
    times until the change of likelihood or lower bound is less than
    ``tol``, otherwise, a ``ConvergenceWarning`` is raised.
    If ``warm_start`` is ``True``, then ``n_init`` is ignored and a single
    initialization is performed upon the first call. Upon consecutive
    calls, training starts where it left off.

    Parameters
    ----------
    X : double[n_samples][n_features]
        List of n_features-dimensional data podoubles. Each row
        corresponds to a single data podouble.

    Returns
    -------	
    labels : int[n_samples]

	*/
	if(this->n_features != n_features || this->n_samples != n_samples){
		cerr<<"Data does not fit"<<endl;
		return;
	}
	this->train_predict(X,labels);
}
void GaussianMixture::train_predict(double **X  ,int* labels){
	/*
	Estimate model parameters using X and predict the labels for X.

    The method trains the model n_init times and sets the parameters with
    which the model has the largest likelihood or lower bound. Within each
    trial, the method iterates between E-step and M-step for `max_iter`
    times until the change of likelihood or lower bound is less than
    `tol`, otherwise, a `ConvergenceWarning` is raised. After trainting, it
    predicts the most probable label for the input data podoubles.

    Parameters
    ----------
    X : double[n_samples][n_features]
        List of n_features-dimensional data podoubles. Each row
        corresponds to a single data podouble.

    Returns
    -------
    labels : double[n_samples]
        Component labels.
	*/
	check_x(this->n_samples,this->n_components);
	this->check_initial_parameters();
	bool do_init = !(this->warm_start);
	int n_init;
	if(do_init)
		n_init = this->n_init;
	else 
		n_init = 1;
	double max_lower_bound = -INFTY;
	this->converged_ = false;
	int rndseed = this->rndseed;
	int init = 0;
	int best_n_iter;
	double log_prob_norm;
	double prev_lower_bound;
	double change;
	double **log_resp =new double*[this->n_samples];
	for(int i=0;i<n_samples;i++)
		log_resp[i] = new double[this->n_components];
	for (init = 0; init < n_init; init++){
		this->print_verbose_msg_init_beg(init);
		if(do_init)
			this->initialize_parameters(X);
		double lower_bound;
		if(do_init)
			lower_bound = -INFTY;
		else
			lower_bound = this->lower_bound_;
		int n_iter = 1;
		for (n_iter = 1; n_iter < this->max_iter + 1; n_iter++){
			prev_lower_bound = lower_bound;
			this->e_step(X,log_prob_norm,log_resp);
			this->m_step(X, log_resp);
			lower_bound = this->compute_lower_bound(log_resp, log_prob_norm);
			change = lower_bound - prev_lower_bound;
            this->print_verbose_msg_iter_end(n_iter, change);
            if (abs(change) < this->tol){
                this->converged_ = true;
                break;
			}
		}
		this->print_verbose_msg_init_end(lower_bound);
		if (lower_bound > max_lower_bound){
            max_lower_bound = lower_bound;

            best_n_iter = n_iter;
		}
	}
	this->e_step(X,log_prob_norm,log_resp);
	if (!this->converged_)
		cerr<<"Initialization "<<(init+1)<<" did not converge. Try different init parameters, or increase max_iter, tol or check for degenerate data."<<endl;

	this->n_iter_ = best_n_iter;
	this->lower_bound_ = max_lower_bound;
	argmax(log_resp,labels,this->n_samples,this->n_components);
	for(int i=0;i<this->n_samples;i++)
		delete[] log_resp[i];
	delete[] log_resp;
}
void GaussianMixture::predict_proba(double **X, double** probs, int n_samples, int n_features){
    /*
	Predict posterior probability of each component given the data.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    a : array, shape (n_samples, n_components)
        Returns the probability each Gaussian (state) in
        the model given each sample.
    */
	if(this->n_features != n_features || this->n_samples != n_samples){
		cerr<<"Data does not fit"<<endl;
		return;
	}
	if(this->X == NULL){
		cerr<<"System not trained. Please use the train() function first."<<endl;
		return;
	}
	check_x(this->n_samples,this->n_components);
	double* log_prob = new double[this->n_samples];
	double** log_resp = new double*[this->n_samples];
	for(int i=0;i<this->n_samples;i++)
		log_resp[i] = new double[this->n_components];
	this->estimate_log_prob_resp(X,log_prob,log_resp);
	npexp(log_resp,probs,this->n_samples,this->n_components);
	for(int i=0;i<this->n_samples;i++)
		delete[] log_resp[i];
	delete[] log_resp;
	delete[] log_prob;
}
void GaussianMixture::check_initial_parameters(){
	/*
	Check values of the basic parameters.

    Parameters
    ----------
    X : double[n_samples][n_features]
	*/
	if(this->n_components <1){
		cerr<<"Invalid value for 'n_components': "<<this->n_components<<"   Estimation requires at least one component";
		exit(1);}
	if(this->tol <0.){
		cerr<<"Invalid value for 'tol': "<<this->tol<<" Tolerance used by the EM must be non-negative";
		exit(1);}
	if(this->n_init <1){
		cerr<<"Invalid value for 'n_init': "<<this->n_init<<" Estimation requires at least one run";
		exit(1);}
	if(this->max_iter <1){
		cerr<<"Invalid value for 'max_iter': "<<this->max_iter<<" Estimation requires at least one iteration";
		exit(1);}
	if(this->reg_covar <0.){
		cerr<<"Invalid value for 'reg_covar': "<<this->reg_covar<<" regularization on covariance must be non-negative";
		exit(1);}
	//Check all the parameters values of the derived class
	this->check_parameters();
}
void GaussianMixture::check_parameters(){
	/*
	Check the Gaussian mixture parameters are well defined.

    Parameters
    ----------
    X : double[n_samples][n_features]
	*/
	if(!(this->covariance_type == "spherical" ||
		this->covariance_type == "tied" ||
		this->covariance_type == "diag" ||
		this->covariance_type == "full")){
		cerr<<"Invalid value for 'covariance_type': "<<this->covariance_type<<" should be in ['spherical', 'tied', 'diag', 'full']"<<endl;
		exit(1);
	}
	if (this->weights_init != NULL)
        check_weights(this->weights_init,this->n_components);
	if (this->precisions_init_full != NULL ||
		this->precisions_init_diag != NULL ||
		this->precisions_init_spherical != NULL ||
		this->precisions_init_tied != NULL)
		if (this->covariance_type == "spherical")
			check_precision1(this->precisions_init_spherical,this->covariance_type,this->n_components);
		else if (this->covariance_type == "tied")
			check_precision2t(this->precisions_init_tied,this->covariance_type,this->n_features);
		else if (this->covariance_type == "diag")
			check_precision2d(this->precisions_init_diag,this->covariance_type,this->n_components,this->n_features);
		else if (this->covariance_type == "full")
			check_precision3(this->precisions_init_full,this->covariance_type,this->n_components,this->n_features);
}
	
void GaussianMixture::initialize_parameters(double **X){
	/*
	Initialize the model parameters.

    Parameters
    ----------
    X : double[n_samples][n_features]

    rndseed : int
        A random number generator instance.
	*/
	double **resp;
	int* label;
	double* resp1;
	if(this->init_params == "kmeans"){
		resp = new double*[this->n_samples];
		for(int i=0;i<this->n_samples;i++)
			resp[i] = new double[this->n_components]; 
		for(int i=0;i<this->n_samples;i++)
			for(int j=0;j<this->n_components;j++)
				resp[i][j]=0;
		label = new int[this->n_samples];
//=================================================================================================================
		getKMeansLabels(this->n_components,X,label,this->max_iter,n_samples,n_features,rndseed);
		//getLabels(n_samples,label,this->n_components,this->rndseed);
		/*	
		KMeans kmeans = KMeans(this->n_features, this->n_components);

		kmeans.Initialize(this->n_samples, X);
		while (kmeans.Cluster(this->n_samples, X)>this->tol);

		for (int i = 0; i < this->n_samples; i++)
			label[i]=kmeans.Classify(X[i]);*/
//=================================================================================================================
		for (int i = 0; i < this->n_samples; i++)
			resp[i][label[i]]=1;
		delete[] label;
	}
	else if (this->init_params == "random"){
		srand(time(0));
		resp = new double*[this->n_samples];
		for(int i=0;i<this->n_samples;i++){
			resp[i] = new double[this->n_components];
			for(int j=0;j<this->n_components;j++)
				resp[i][j]=rand()%(2);
		}
		resp1=new double[this->n_samples];
		npsum(resp,resp1,this->n_samples,this->n_components);
		for(int i=0;i<this->n_samples;i++)
			for(int j=0;j<this->n_components;j++)
				resp[i][j] /= resp1[i];
		delete[] resp1;
	}
	else{
		cerr<<"Unimplemented initialization method "<<this->init_params<<endl;
		exit(0);
	}
	this->initialize(X,resp);
	for(int i=0;i<n_samples;i++)
			delete[] resp[i];
	delete[] resp;
		
}
void GaussianMixture::initialize(double **X,double **resp){
	/*
	Initialize the model parameters of the derived class.

    Parameters
    ----------
    X : double[n_samples][n_features]

    resp : double[n_samples][n_components]
	*/
	double *weights = new double[this->n_components] ;
	estimate_gaussian_covariances_nk(resp,weights, this->n_components,this->n_samples);
	double **means = new double*[this->n_components];
	for(int i=0;i<this->n_components;i++)
		means[i] = new double[this->n_features];
	estimate_gaussian_covariances_means(resp,X,weights,means,this->n_components,this->n_features,this->n_samples) ;
	
	double *covariances_spherical;
	double	**covariances_tied;
	double	**covariances_diag;
	double	***covariances_full;
		
	if (this->covariance_type == "spherical")
		covariances_spherical = new double[this->n_components];
	else if (this->covariance_type == "tied"){
		covariances_tied= new double*[this->n_features];
		for(int j=0;j<this->n_features;j++)
			covariances_tied[j] = new double[this->n_features];
	}
	else if (this->covariance_type == "diag"){
		covariances_diag= new double*[this->n_components];
		for(int i=0;i<this->n_components;i++)
			covariances_diag[i] = new double[this->n_features];
	}
	else if (this->covariance_type == "full"){
		covariances_full= new double**[this->n_components];
		for(int i=0;i<this->n_components;i++){
			covariances_full[i] = new double*[this->n_features];
			for(int j=0;j<this->n_features;j++)
				covariances_full[i][j] = new double[n_features];
		}
	}
	if (this->covariance_type == "spherical")
		estimate_gaussian_covariances_spherical(resp,X,weights,means,covariances_spherical,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "tied")
		estimate_gaussian_covariances_tied(resp,X,weights,means,covariances_tied,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "diag")
		estimate_gaussian_covariances_diag(resp,X,weights,means,covariances_diag,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "full")
		estimate_gaussian_covariances_full(resp,X,weights,means,covariances_full,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	
	for(int i=0;i<this->n_components;i++)
		weights[i] /= this->n_samples;
	
	if(this->weights_init == NULL){
		if(this->warm_start)
			delete [] this->weights_;
		this->weights_ = weights;
	}else{
		this->weights_ = this->weights_init;
		delete [] weights;
	}if(this->means_init == NULL){
		if(this->warm_start){
			for(int i=0;i<this->n_components;i++)
				delete [] this->means_[i];
			delete [] this->means_;
		}
		this->means_ = means;
	}else{
		this->means_ = this->means_init;
		for(int i=0;i<this->n_components;i++)
			delete [] means[i];
		delete [] means;
	}
	
	if (this->precisions_init_spherical == NULL &&
		this->precisions_init_tied == NULL &&
		this->precisions_init_diag == NULL &&
		this->precisions_init_full == NULL){
		if (this->covariance_type == "spherical"){
			if(this->warm_start){
				delete [] this->covariances_spherical_;
			}else{
				this->precisions_cholesky_spherical_ = new double[this->n_components];
			}
			this->covariances_spherical_ = covariances_spherical;
			compute_precision_cholesky_spherical(covariances_spherical,this->precisions_cholesky_spherical_,this->n_components);
		}else if (this->covariance_type == "tied"){
			if(this->warm_start){
				for(int j=0;j<this->n_features;j++)
					delete [] this->covariances_tied_[j];
				delete [] this->covariances_tied_;
			}else{
				this->precisions_cholesky_tied_= new double*[this->n_features];
				for(int j=0;j<this->n_features;j++)
					this->precisions_cholesky_tied_[j] = new double[this->n_features];
			}
			this->covariances_tied_ = covariances_tied;
			compute_precision_cholesky_tied(covariances_tied,this->precisions_cholesky_tied_,this->n_features);
		}else if (this->covariance_type == "diag"){
			if(this->warm_start){
				for(int i=0;i<this->n_components;i++)
					delete [] this->covariances_diag_[i];
				delete [] this->covariances_diag_;
			}else{
				this->precisions_cholesky_diag_ = new double*[this->n_components];
				for(int i=0;i<this->n_components;i++)
					this->precisions_cholesky_diag_ [i] = new double[this->n_features];
			}
			this->covariances_diag_ = covariances_diag;
			compute_precision_cholesky_diag(covariances_diag,this->precisions_cholesky_diag_,this->n_components,this->n_features);
		}else if (this->covariance_type == "full"){
			if(this->warm_start){
				for(int i=0;i<this->n_components;i++){
					for(int j=0;j<this->n_features;j++)
						delete [] this->covariances_full_[i][j];
					delete [] this->covariances_full_[i];
				}
				delete [] this->covariances_full_;
			}else{
				this->precisions_cholesky_full_= new double**[this->n_components];
				for(int i=0;i<this->n_components;i++){
					this->precisions_cholesky_full_[i] = new double*[n_features];
					for(int j=0;j<n_features;j++)
						this->precisions_cholesky_full_[i][j] = new double[n_features];
				}
			}
			this->covariances_full_ = covariances_full;			
			compute_precision_cholesky_full(covariances_full,this->precisions_cholesky_full_,this->n_components,this->n_features);
		}
	}else if (this->covariance_type == "full"){
		this->precisions_cholesky_full_ = this->precisions_init_full;
		for(int i=0;i<this->n_components;i++){
			for(int j=0;j<this->n_features;j++)
				delete [] covariances_full[i][j];
			delete [] covariances_full[i];
		}
		delete [] covariances_full;
	}else if (this->covariance_type == "tied"){
		linalg_cholesky(this->precisions_init_tied,this->precisions_cholesky_tied_,this->n_features) ;
		for(int j=0;j<this->n_features;j++)
			delete [] covariances_tied[j];
		delete [] covariances_tied;
	}else if (this->covariance_type == "diag"){
		this->precisions_cholesky_diag_  = this->precisions_init_diag ;
		for(int i=0;i<this->n_components;i++)
			delete [] covariances_diag[i];
		delete [] covariances_diag;
	}else if (this->covariance_type == "spherical"){
		this->precisions_cholesky_spherical_  = this->precisions_init_spherical ;
		delete [] covariances_spherical;
	}
	this->X = X;
	this->warm_start = true;
}

void GaussianMixture::e_step(double **X,double &log_prob_norm,double**log_responsibility){
	double* log_prom = new double[this->n_samples];
	this->estimate_log_prob_resp(X,log_prom,log_responsibility);
	log_prob_norm = mean(log_prom,this->n_samples);
	delete[] log_prom;
}

void GaussianMixture::estimate_log_prob_resp(double **X,double* log_prob_norm,double**log_responsibility){
	double** weighted_log_prob = new double*[this->n_samples];
	for(int i=0;i<this->n_samples;i++)
		weighted_log_prob[i] = new double[this->n_components];
	this->estimate_weighted_log_prob(X,weighted_log_prob);
	logsumexp(weighted_log_prob,log_prob_norm,this->n_samples,this->n_components);
	for (int i = 0; i < this->n_samples; i++)
		for (int j = 0; j < this->n_components; j++)
			log_responsibility[i][j] = weighted_log_prob[i][j] - log_prob_norm[i];
	for(int i=0;i<this->n_samples;i++)
		delete[] weighted_log_prob[i];
	delete[] weighted_log_prob;
}
void GaussianMixture::estimate_weighted_log_prob(double **X,double** a){
	double** log_prob = new double*[this->n_samples];
	double* log_weights = new double[this->n_components];
	for(int i=0;i<this->n_samples;i++)
		log_prob[i] = new double[this->n_components];
	this->estimate_log_prob(X,log_prob);
	this->estimate_log_weights(log_weights);
	summatrixarray(log_prob,log_weights,a,this->n_samples,this->n_components);
	for(int i=0;i<this->n_samples;i++)
		delete[] log_prob[i];
	delete[] log_weights;
	delete[] log_prob;
}
void GaussianMixture::estimate_log_prob(double **X,double** log_prob){
	if (this->covariance_type == "spherical")
		log_gaussian_prob_spherical(X,this->means_,this->precisions_cholesky_spherical_,log_prob,this->n_samples,this->n_components,this->n_features);
	else if (this->covariance_type == "tied")
		log_gaussian_prob_tied(X,this->means_,this->precisions_cholesky_tied_,log_prob,this->n_samples,this->n_components,this->n_features);
	else if (this->covariance_type == "diag")
		log_gaussian_prob_diag(X,this->means_,this->precisions_cholesky_diag_,log_prob,this->n_samples,this->n_components,this->n_features);
	else if (this->covariance_type == "full")
		log_gaussian_prob_full(X,this->means_,this->precisions_cholesky_full_,log_prob,this->n_samples,this->n_components,this->n_features);
}
void GaussianMixture::estimate_log_weights(double *we){
		nplog(this->weights_,we,this->n_components);
}

void GaussianMixture::m_step(double**X,double** log_resp){
	/*
	M step.

    Parameters
    ----------
    X : double[n_samples][n_features]

    log_resp : double[n_samples][n_components]
        Logarithm of the posterior probabilities (or responsibilities) of
        the point of each sample in X.
	*/
	double** resp = new double*[this->n_samples];
	for (int i = 0; i < this->n_samples; i++)
		resp[i] = new double[this->n_components];
	npexp(log_resp,resp,this->n_samples,this->n_components);
	estimate_gaussian_covariances_nk(resp,this->weights_,this->n_components,this->n_samples);
	estimate_gaussian_covariances_means(resp,X,this->weights_,this->means_,this->n_components,this->n_features,this->n_samples);
	if (this->covariance_type == "spherical")
		estimate_gaussian_covariances_spherical(resp,X,this->weights_,this->means_,this->covariances_spherical_,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "tied")
		estimate_gaussian_covariances_tied(resp,X,this->weights_,this->means_,this->covariances_tied_,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "diag")
		estimate_gaussian_covariances_diag(resp,X,this->weights_,this->means_,this->covariances_diag_,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	else if (this->covariance_type == "full")
		estimate_gaussian_covariances_full(resp,X,this->weights_,this->means_,this->covariances_full_,this->reg_covar,this->n_components,this->n_features,this->n_samples);
	for(int i=0;i<this->n_components;i++)
		this->weights_[i] /= this->n_samples;
	if (this->covariance_type == "spherical")
			compute_precision_cholesky_spherical(this->covariances_spherical_,this->precisions_cholesky_spherical_,this->n_components);
	else if (this->covariance_type == "tied")
		compute_precision_cholesky_tied(this->covariances_tied_,this->precisions_cholesky_tied_,this->n_features);
	else if (this->covariance_type == "diag")
		compute_precision_cholesky_diag(this->covariances_diag_,this->precisions_cholesky_diag_,this->n_components,this->n_features);
	else if (this->covariance_type == "full")
		compute_precision_cholesky_full(this->covariances_full_,this->precisions_cholesky_full_,this->n_components,this->n_features);
	for (int i = 0; i < this->n_samples; i++)
		delete[] resp[i];
	delete[] resp;
}

double GaussianMixture::compute_lower_bound(double** _,double log_prob_norm){
	return log_prob_norm;
}

void GaussianMixture::print_verbose_msg_iter_end(int n_iter,double diff_ll){
    if (n_iter % this->verbose_interval == 0)
        if (this->verbose == 1)
            cout<<"  Iteration "<< n_iter<<endl;
        else if (this->verbose >= 2){
			time_t cur_time = time(NULL);
			cout<<"  Iteration "<<n_iter<<"\t time lapse "<<(cur_time - this->iter_prev_time)<<"\t ll change "<<diff_ll<<endl;
            this->iter_prev_time = cur_time;
		}
}
void GaussianMixture::print_verbose_msg_init_end(double ll){
    if (this->verbose == 1)
        cout<<"Initialization converged: "<<this->converged_<<endl;
    else if (this->verbose >= 2)
        cout<<"Initialization converged: "<<this->converged_<<"\t time lapse "<<(time(NULL) -this->init_prev_time)<<"\t ll "<< ll;
}
void GaussianMixture::print_verbose_msg_init_beg(int n_init){
    /*Print verbose message on initialization.*/
    if (this->verbose == 1)
        cout<<"Initialization "<<n_init<<endl;
    else if (this->verbose >= 2){
        cout<<"Initialization "<<n_init<<endl;
        this->init_prev_time = time(NULL);
        this->iter_prev_time = this->init_prev_time;
	}
}

void GaussianMixture::get_parameters_spherical_(double*&weights,double**&means,double*&covariances) {
	weights = this->weights_;
	means = this->means_;
	covariances = this->covariances_spherical_;
}
void GaussianMixture::get_parameters_tied_(double*&weights,double**&means,double**&covariances) {
	weights = this->weights_;
	means = this->means_;
	covariances = this->covariances_tied_;
}
void GaussianMixture::get_parameters_diag_(double*&weights,double**&means,double**&covariances) {
	weights = this->weights_;
	means = this->means_;
	covariances = this->covariances_diag_;
}
void GaussianMixture::get_parameters_full_(double*&weights,double**&means,double***&covariances) {
	weights = this->weights_;
	means = this->means_;
	covariances = this->covariances_full_;
}
	
void GaussianMixture::set_parameters_spherical_(double*&weights,double**&means,double*&covariances,double*&precisions_cholesky,int n_features) {
	this->weights_ = weights;
	this->means_ = means;
	this->covariances_spherical_ = covariances;
	this->precisions_cholesky_spherical_ = precisions_cholesky;
	this->set_parameters();
}
void GaussianMixture::set_parameters_tied_(double*&weights,double**&means,double**&covariances,double**&precisions_cholesky,int n_features) {
	this->weights_ = weights;
	this->means_ = means;
	this->covariances_tied_ = covariances;
	this->precisions_cholesky_tied_ = precisions_cholesky;
	this->set_parameters();
}
void GaussianMixture::set_parameters_diag_(double*&weights,double**&means,double**&covariances,double**&precisions_cholesky,int n_features) {
	this->weights_ = weights;
	this->means_ = means;
	this->covariances_diag_ = covariances;
	this->precisions_cholesky_diag_ = precisions_cholesky;
	this->set_parameters();
}
void GaussianMixture::set_parameters_full_(double*&weights,double**&means,double***&covariances,double***&precisions_cholesky,int n_features) {
	this->weights_ = weights;
	this->means_ = means;
	this->covariances_full_ = covariances;
	this->precisions_cholesky_full_ = precisions_cholesky;
	this->set_parameters();
}
	
void GaussianMixture::set_parameters(){
	double** prec_cholT = new double*[this->n_features];
	for(int i=0;i<this->n_features;i++)
		prec_cholT[i] = new double[this->n_features];
    if (this->covariance_type == "full"){
		this->precisions_full_= new double**[this->n_components];
		for(int i=0;i<this->n_components;i++){
			this->precisions_full_[i] = new double*[this->n_features];
			for(int j=0;j<this->n_features;j++)
				this->precisions_full_[i][j] = new double[this->n_features];
		}
		for (int k = 0; k < this->n_components; k++){
			trans_matrix(this->precisions_cholesky_full_[k],prec_cholT,this->n_features,this->n_features);
			dotequ(this->precisions_cholesky_full_[k],prec_cholT,this->precisions_full_[k],this->n_features,this->n_features,this->n_features);
		}
	}else if (this->covariance_type == "tied"){
		this->precisions_tied_= new double*[this->n_features];
		for(int i=0;i<this->n_features;i++)
			this->precisions_tied_[i] = new double[this->n_features];
		trans_matrix(this->precisions_cholesky_tied_,prec_cholT,this->n_features,this->n_features);
		dotequ(this->precisions_cholesky_tied_,prec_cholT,this->precisions_tied_ ,this->n_features,this->n_features,this->n_features);
	}else if (this->covariance_type == "spherical"){
		this->precisions_spherical_= new double[this->n_components];
		for (int i = 0; i < this->n_components; i++)
			this->precisions_spherical_[i] = this->precisions_cholesky_spherical_[i] * this->precisions_cholesky_spherical_[i];
	}else if (this->covariance_type == "diag"){
		this->precisions_diag_ = new double*[this->n_components];
		for (int i = 0; i < this->n_components; i++){
			this->precisions_diag_[i] = new double[this->n_features];
			for (int j = 0; j < this->n_features; j++)
				this->precisions_diag_[i][j] = this->precisions_cholesky_diag_[i][j] * this->precisions_cholesky_diag_[i][j];
		}
	}
	for(int i=0;i<this->n_features;i++)
		delete[] prec_cholT[i];
	delete[] prec_cholT;
}

int GaussianMixture::n_parameters(){
    /*Return the number of free parameters in the model.*/
	double cov_params;
    if (this->covariance_type == "full")
        cov_params = this->n_components * this->n_features * (this->n_features + 1) / 2.;
    else if (this->covariance_type == "diag")
        cov_params = this->n_components * this->n_features;
    else if (this->covariance_type == "tied")
        cov_params = this->n_features * (this->n_features + 1) / 2.;
    else if (this->covariance_type == "spherical")
        cov_params = this->n_components;
    double mean_params = this->n_features * this->n_components;
    return int(cov_params + mean_params + this->n_components - 1);
}

double GaussianMixture::bic(){
	/*
	Bayesian information criterion for the current model on the input X.

    Parameters
    ----------
    X : double[n_samples][n_dimensions]

    Returns
    -------
    bic : double
        The lower the better.
	*/
	return (-2 * this->log_likelihood() * this->n_samples + this->n_parameters() * log(this->n_samples));
}
double GaussianMixture::aic(){
	/*
	Akaike information criterion for the current model on the input X.

    Parameters
    ----------
    X : double[n_samples][n_dimensions]

    Returns
    -------
    aic : double
        The lower the better.
	*/
	return -2 * this->log_likelihood() * this->n_samples + 2 * this->n_parameters();
}

double GaussianMixture::log_likelihood(){
	/*Compute the per-sample average log-likelihood of the given data X.

    Parameters
    ----------
    X : double[n_samples][n_dimensions]
        List of n_features-dimensional data points. Each row
        corresponds to a single data point.

    Returns
    -------
    log_likelihood : double
		Log likelihood of the Gaussian mixture given X.
	*/
	double* s = new double[this->n_samples];
	this->score_samples(s);
	double a = mean(s,this->n_samples);
	delete[] s;
	return a;
} 

void GaussianMixture::score_samples(double* s){
	check_x(this->n_samples,this->n_components);
	double** a = new double*[this->n_samples];
	for(int i=0;i<this->n_samples;i++)
		a[i] = new double[this->n_components];
	this->estimate_weighted_log_prob(this->X,a);
	logsumexp(a,s,this->n_samples,this->n_components);
	for(int i=0;i<this->n_samples;i++)
		delete[] a[i];
	delete[] a;
}
