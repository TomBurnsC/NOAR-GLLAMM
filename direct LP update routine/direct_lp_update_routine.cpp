


bool ocgllamm::is_fixed_effect_param(int idx) {

/*
  Helper function that checks whether modified params belong to fixed effects
*/
    return idx >= 0 &&
           static_cast<unsigned int>(idx) < num_parameters &&
           wt_coef[idx] == 0 &&
           cat_lv_inds[idx] < 0;
}


bool ocgllamm::constraints_for_param_are_fixed_effects(int idx) {

/*
  Helper function that checks whether modified params have associated variable constraints 
*/

    if (idx < 0) {
        return true;
    }

    for (unsigned int k = 0; k < constraints.size(); k++) {
        int constrained_idx = -1;

        if (constraints[k].index_1 == idx &&
            constraints[k].index_2 != -1) {
            constrained_idx = constraints[k].index_2;
        }


        else if (constraints[k].index_2 == idx) {
            constrained_idx = constraints[k].index_1;
        }
        if (constrained_idx != -1 &&
            !is_fixed_effect_param(constrained_idx)) {
            return false;
        }
    }

    return true;
}

double ocgllamm::cl_calc_mod_log_likelihood(int i, double delta1, int j, double delta2) {
  std::chrono::time_point<std::chrono::steady_clock> start, stop;
  std::vector<std::chrono::duration<long long, std::nano>> intervals;
  std::vector<unsigned int> reps;
  vector<bool> cat_done;
  
    bool i_is_fixed_effect = is_fixed_effect_param(i);
    
    bool j_is_fixed_effect =
        j < 0 || is_fixed_effect_param(j);
    
    bool i_constraints_are_fixed =
        constraints_for_param_are_fixed_effects(i);
    
    bool j_constraints_are_fixed =
        j < 0 || constraints_for_param_are_fixed_effects(j);


// diverts log likelihood calculation routine if parameter pair values are both fixed effects
    
    if (i_is_fixed_effect &&
        j_is_fixed_effect &&
        i_constraints_are_fixed &&
        j_constraints_are_fixed) {
    
        return cl_calc_orig_mod_log_likelihood(
            i,
            delta1,
            j,
            delta2
        );
    }
      double oc1, oc2;
      vector<int> counters;
      for(unsigned int k = 0; k < num_latent; k++) {
        counters.push_back(0);
      }
    
      for(unsigned int k = 0; k < levels; k++) {
        reps.push_back(1);
        cat_done.push_back(false);
      }

  //  start = std::chrono::steady_clock::now();    
  oc1 = gsl_matrix_get(all_coefs, i, 0);
  oc2 = 0;
  if (wt_coef[i] == 0) {
    *(all_coef_ptrs[i]) = oc1 + delta1;
    int lvi = cat_lv_inds[i];
    if (lvi > -1) {
      int counter = i - 1;
      while((counter > 0) && (cat_lv_inds[counter] == lvi)) {
	counter--;
      }
      int pt_no = i - 1 - counter;
      gsl_matrix_set(point_matrices[0], lvi, pt_no, oc1 + delta1);
      coefs_to_points_and_weights(i); 
    }
  }
  else {
    gsl_matrix_set(all_coefs, i, 0, oc1+delta1);
    coefs_to_points_and_weights(i); 
  }
  if (j > -1) {
    if (i == j) {
      oc2 = oc1 + delta1;
    }
    else {
      oc2 = gsl_matrix_get(all_coefs, j, 0);
    }
    if (wt_coef[j] == 0) {
      *(all_coef_ptrs[j]) = oc2 + delta2;
      int lvi = cat_lv_inds[j];
      if (lvi > -1) {
        int counter = j - 1;
        while((counter > 0) && (cat_lv_inds[counter] == lvi)) {
          counter--;
        }
	int pt_no = j - 1 - counter;
	gsl_matrix_set(point_matrices[0], lvi, pt_no, oc2 + delta2);
	coefs_to_points_and_weights(j); 
      }
    }
    else {
      gsl_matrix_set(all_coefs, j, 0, oc2+delta2);
      coefs_to_points_and_weights(j); 
    }
  }
  //  stop  = std::chrono::steady_clock::now();
  //  std::cout << "Time taken to adjust coefficients = " << (stop-start).count() << " ns" << std::endl;
  //  start  = std::chrono::steady_clock::now();
  
  if (constraints.size() > 0) {
    for (unsigned int k = 0; k < constraints.size(); k++) {
      if ((constraints[k].index_1 == i) &&
	  (constraints[k].index_2 != -1)){
	*all_coef_ptrs[constraints[k].index_2] = *all_coef_ptrs[i];
      }
      if (constraints[k].index_2 == i) {
	*all_coef_ptrs[constraints[k].index_1] = *all_coef_ptrs[i];
      }
      if ((constraints[k].index_1 == j)  &&
	  (constraints[k].index_2 != -1)){
	*all_coef_ptrs[constraints[k].index_2] = *all_coef_ptrs[j];
      }
      if (constraints[k].index_2 == j) {
	*all_coef_ptrs[constraints[k].index_1] = *all_coef_ptrs[j];
      }
    }
  }

  
  // double nl = calc_log_likelihood();
  if ((cat_lv_inds[i] > -1) || ((j > -1) && (j != i) && (cat_lv_inds[j] > -1))) {
    //    calc_last_point_and_weight();
    
    for (auto p : lwts) {
      delete p;
    }
    lwts.clear();

    for (unsigned int level = 0; level < levels - 1; level++) {
      reps[level] = 1;
      cat_done.at(level) = false;
      double_vec* wvec = new double_vec(points_per_level[level], 1.0);
      for(unsigned int lv = 0; lv < num_latent; lv++) {
	if (latent_info[lv].level >= level+2) {
	  unsigned int pt = 0;
	  unsigned int rep = 0;
	  for (int col = 0; col < points_per_level[level]; col++) {
	    if((latent_info[lv].continuous == true) || (cats_per_level == false) || (cat_done.at(level) == false)) {
	      wvec->at(col) *= qweights.at(lv)->at(pt);
	      rep++;
	      if (rep >= reps[level]) {
		pt++;
		rep = 0;
		if (pt >= latent_info[lv].num_points) {
		  pt = 0;
		}
	      }
	    }
	  }
	  if ((cats_per_level == false) ||
	      (latent_info[lv].continuous == true)) {
            reps[level] *= latent_info[lv].num_points;
	  }
	  else {
	    cat_done.at(level) = true;
	  }
      
	}
      }
      lwts.push_back(wvec);
    }

    if (use_opencl) {
      
      for (size_t i = 0; i < point_matrices.size(); i++) {
	//    DLOG(INFO) << "About to copy point_matrices to cl_point_matrices";
	copy_matrix_to_ocl(point_matrices[i], cl_point_matrices[i], true,  CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
	//    DLOG(INFO) << "Copied point_matrices to cl_point_matrices";
	cl::Buffer* temp = nullptr;
	if (cl_lwts.size() > 0) {
	  for (auto p : cl_lwts) {
	    delete p;
	  }
	  cl_lwts.clear();
	}
	for(unsigned int i = 0; i  <levels-1; i++) {
	  temp = new cl::Buffer(*ocl_ctx, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
				lwts[i]->size()*sizeof(cl_double));
	  //    DLOG(INFO) << "Created new cl::buffer for copying of size " << lwts[i]->size()*sizeof(cl_double);
	  ocl_q->enqueueWriteBuffer(*temp, CL_TRUE, 0, (lwts[i]->size()*sizeof(cl_double)), lwts[i]->data());
	  ocl_q->finish();
	  cl_lwts.insert(cl_lwts.end(), temp);
	  //    DLOG(INFO) << "Added cl::Buffer to cl_lwts";
	}
      }

  
    }
  }
  copy_matrix_to_ocl(beta, cl_beta, true, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
  if (levels > 1) {
    copy_matrix_to_ocl(lambda, cl_lambda, true, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR);
  }     
  lp_valid = false;
  clp_valid = false;
  cllp_valid = false;
  llp_valid = false;
  cpv_valid  = false;
  double nl = cl_calculate_log_likelihood();
  //  cl_dbuffer_output(cl_latent_linear_predictor, num_observations, tot_points, 2, 24);
  //  stop  = std::chrono::steady_clock::now();
  //  std::cout << "Time taken to calculate log likelihood = " << (stop-start).count() << " ns" << std::endl;
  //  start  = std::chrono::steady_clock::now();
  //gsl_matrix_output(beta);
  //if (lambda != nullptr) {
  //  gsl_matrix_output(lambda);
  //}
  // cout << "ll -> " << nl << endl;
  cpv_valid = false;
  clp_valid = false;
  cllp_valid = false;
  lp_valid = false;
  llp_valid = false;
  if (j > -1) {
    if (wt_coef[j] == 0) {
      *(all_coef_ptrs[j]) = oc2;
      if (cat_lv_inds[j] > -1) {
        coefs_to_points_and_weights(j); 
      }
    }
    else {
      gsl_matrix_set(all_coefs, j, 0, oc2);
      coefs_to_points_and_weights(j); 
    }
  }
  if (wt_coef[i] == 0) {
    *(all_coef_ptrs[i]) = oc1;
    if (cat_lv_inds[i] > -1) {
      coefs_to_points_and_weights(i); 
    }
  }
  else {
    gsl_matrix_set(all_coefs, i, 0, oc1);
    coefs_to_points_and_weights(i); 
  }    
  //  stop  = std::chrono::steady_clock::now();
  //  std::cout << "Time taken to reset coefficients = " << (stop-start).count() << " ns" << std::endl;
  //  start  = std::chrono::steady_clock::now();
  if ((cat_lv_inds[i] > -1) || ((j > -1) && (j != i) && (cat_lv_inds[j] > -1))) {
    //    calc_last_point_and_weight();
    
    for (auto p : lwts) {
      delete p;
    }
    lwts.clear();

    for (unsigned int level = 0; level < levels - 1; level++) {
      reps[level] = 1;
      cat_done.at(level) = false;
      double_vec* wvec = new double_vec(points_per_level[level], 1.0);
      for(unsigned int lv = 0; lv < num_latent; lv++) {
	if (latent_info[lv].level >= level+2) {
	  unsigned int pt = 0;
	  unsigned int rep = 0;
	  for (int col = 0; col < points_per_level[level]; col++) {
	    if((latent_info[lv].continuous == true) || (cats_per_level == false) || (cat_done.at(level) == false)) {
	      wvec->at(col) *= qweights.at(lv)->at(pt);
	      rep++;
	      if (rep >= reps[level]) {
		pt++;
		rep = 0;
		if (pt >= latent_info[lv].num_points) {
		  pt = 0;
		}
	      }
	    }
	  }
	  if ((cats_per_level == false) ||
	      (latent_info[lv].continuous == true)) {
            reps[level] *= latent_info[lv].num_points;
	  }
	  else {
	    cat_done.at(level) = true;
	  }
      
	}
      }
      lwts.push_back(wvec);
    }
  
  }

  return nl;
}



double ocgllamm::cl_calc_orig_mod_log_likelihood(
    int i,
    double delta1,
    int j,
    double delta2
) {

/*
  This function  handles fixed-effects parameters only 
*/
    double result;
    
    // cl_set_linear_predictor(); /lp unchanged on GPU

    cl_modify_linear_predictors(i, delta1);

    if (j > -1) {
        cl_modify_linear_predictors(j, delta2);
    }

    clp_valid = true;
    cpv_valid = false;

    if (levels > 1) {
        cllp_valid = false;
        cl_set_latent_linear_predictor();
    }

    result = cl_calculate_log_likelihood();

    cl_modify_linear_predictors(i, -delta1);

    if (j > -1) {
        cl_modify_linear_predictors(j, -delta2);
    }

    clp_valid = true;
    cpv_valid = false;

    if (levels > 1) {
        cllp_valid = false;
    }

    return result;
}


int ocgllamm::cl_modify_linear_predictors(
    unsigned int i,
    double delta)
{
    /*
     * This routine launches the kernel responsible for modifying the ordinary linear predictor
     * according to  fixed-effect contributions.
     */
     
    if (i >= num_parameters) {
        std::cout
            << "Cannot modify linear predictor with coefficient "
            << i
            << ": there are only "
            << num_parameters
            << " fixed-effect coefficients"
            << std::endl;

        return CL_INVALID_VALUE;
    }

    cl::Kernel kernel = mlp_kernel;
    int err = CL_SUCCESS;

    err |= kernel.setArg(0, delta);
    err |= kernel.setArg(1, i);
    err |= kernel.setArg(2, num_parameters);
    err |= kernel.setArg(3, *cl_X);
    err |= kernel.setArg(4, *cl_linear_predictor);

    // The current kernel still has the point_matrix argument,
    // but it is unused when null.
    cl_mem null_mem = nullptr;
    err |= kernel.setArg(5, sizeof(cl_mem), &null_mem);

    if (err != CL_SUCCESS) {
        std::cout
            << "Setting modify_lp arguments failed with error code "
            << err
            << std::endl;

        return err;
    }

    const cl::NDRange global(num_observations, 1);

    err = ocl_q->enqueueNDRangeKernel(
        kernel,
        cl::NullRange,
        global,
        cl::NullRange
    );

    if (err != CL_SUCCESS) {
        std::cout
            << "Modifying linear predictor failed with error code "
            << err
            << std::endl;

        return err;
    }

    ocl_q->finish();

    clp_valid = true;
    cpv_valid = false;

    // The cached latent predictor was derived from the old LP.
    if (levels > 1) {
        cllp_valid = false;
    }

    return CL_SUCCESS;
}
