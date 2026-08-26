int ocl_calc_dlikelihoods(cl::Buffer *obs, cl::Buffer *pred, 
			  cl::Buffer *likelihoods, cl::Buffer *log_factorials,
			  int global_size, unsigned int cols, Family family, double sigma2) {  
  cl::Kernel kernel;
  //  cl_int clerr = 0;
  int err = 0;
  
  switch (family) {
  case Gamma :
  case Gaussian :
    kernel = mdlgauss_kernel;
    break;
  case Binomial :
    kernel = mdlbin_kernel;
    break;
  case Poisson :
    kernel = mdlpois_kernel;
    kernel.setArg(4, *log_factorials);
    break;
  }

  cl::NDRange gndr = cl::NDRange(global_size, cols);
  
  err |= kernel.setArg(0, *obs);
  err |= kernel.setArg(1, *pred);
  err |= kernel.setArg(2, *likelihoods);
  err |= kernel.setArg(3, cols);
  if (family == Gaussian || family == Gamma) {
    
    double gauss_const = -0.5 * log(2.0 * sigma2 * M_PI);
    double inv_2sigma2 = -0.5 / sigma2;
    
    err |= kernel.setArg(4, inv_2sigma2);
    err |= kernel.setArg(5, gauss_const);
  }
  
    cl::Event mdl_event;
    
    err |= ocl_q->enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        gndr,
        cl::NullRange,
        nullptr,
        &mdl_event);
    
    mdl_event.wait();
    
  switch (family) {
  case Gamma :
  case Gaussian : 
    record_kernel_event(mdl_event, mdl_gauss_total, mdl_gauss_calls);
    break;
  case Binomial :
    record_kernel_event(mdl_event, mdl_binom_total, mdl_binom_calls);
    break;
  case Poisson :
    record_kernel_event(mdl_event, mdl_poiss_total, mdl_poiss_calls);
    break; 
  }
  
  if (err != 0) {
    std::cout << "Calculating LIkelihoods failed with error code " << err << std::endl;
  }
  ocl_q->finish();
  return err;
};
