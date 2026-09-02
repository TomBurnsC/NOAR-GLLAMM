bool ocl_initialised = false;
bool kernels_initialised = false;

static double sexp_total      = 0.0;
static double sexpit_total    = 0.0;
static double srecip_total    = 0.0;
static double scll_total      = 0.0;
static double sid_total       = 0.0;

static double dexp_total      = 0.0;
static double dexpit_total    = 0.0;
static double drecip_total    = 0.0;
static double dcll_total      = 0.0;

static double dupper_total    = 0.0;
static double dlower_total    = 0.0;
static double dsub_total      = 0.0;

// Reduction kernels
static double dra_total       = 0.0;
static double drs_total       = 0.0;
static double dssa_total      = 0.0;
static double dsss_total      = 0.0;
static double sra_total       = 0.0;
static double srs_total       = 0.0;
static double sssa_total      = 0.0;
static double ssss_total      = 0.0;

// Likelihood kernels
static double mdl_gauss_total  = 0.0;
static double mdl_binom_total  = 0.0;
static double mdl_poiss_total  = 0.0;


// Latent variable / reduction kernels
static double dllp_total              = 0.0;
static double red_to_s_by_unit_total  = 0.0;
static double red_to_a_by_unit_total  = 0.0;
static double red_by_lv_total         = 0.0;

// Kernel profiling call counters
static unsigned long sexp_calls = 0;
static unsigned long sexpit_calls = 0;
static unsigned long srecip_calls = 0;
static unsigned long scll_calls = 0;
static unsigned long sid_calls = 0;
static unsigned long dexp_calls = 0;
static unsigned long dexpit_calls = 0;
static unsigned long drecip_calls = 0;
static unsigned long dcll_calls = 0;
static unsigned long dupper_calls = 0;
static unsigned long dlower_calls = 0;
static unsigned long dsub_calls = 0;
static unsigned long dra_calls = 0;
static unsigned long drs_calls = 0;
static unsigned long dssa_calls = 0;
static unsigned long dsss_calls = 0;
static unsigned long sra_calls = 0;
static unsigned long srs_calls = 0;
static unsigned long sssa_calls = 0;
static unsigned long ssss_calls = 0;
static unsigned long mdl_gauss_calls = 0;
static unsigned long mdl_binom_calls = 0;
static unsigned long mdl_poiss_calls = 0;
static unsigned long dllp_calls = 0;
static unsigned long red_to_s_by_unit_calls = 0;
static unsigned long red_to_a_by_unit_calls = 0;
static unsigned long red_by_lv_calls = 0;



static cl_ulong record_kernel_event(const cl::Event& event,
                                    double& total_ms,
                                    unsigned long& calls) {
  cl_ulong start =
      event.getProfilingInfo<CL_PROFILING_COMMAND_START>();

  cl_ulong end =
      event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  const cl_ulong runtime_ns = end - start;

  total_ms += static_cast<double>(runtime_ns) / 1e6;
  calls++;

  return runtime_ns;
}

static void print_kernel_profile(const char* name,
                                 double total_ms,
                                 unsigned long calls) {
  std::cout << name
            << " total_ms = " << total_ms
            << ", calls = " << calls;

  if (calls > 0) {
    std::cout << ", average_ms = "
              << (total_ms / static_cast<double>(calls));
  }

  std::cout << std::endl;
}

std::vector<std::vector<cl::Device> > ocl_devices;
cl::Device       *ocl_dev= nullptr;
cl::Context      *ocl_ctx= nullptr;
cl::CommandQueue *ocl_q  = nullptr;
cl_command_queue *ocl_qp = nullptr;
int               ocl_work_group_size = 0;
cl::Kernel        sexp_kernel;
cl::Kernel        sexpit_kernel;
cl::Kernel        srecip_kernel;
cl::Kernel        scll_kernel;
cl::Kernel        sid_kernel;
cl::Kernel        dexpit_kernel;
cl::Kernel        dexp_kernel;
cl::Kernel        drecip_kernel;
cl::Kernel        dcll_kernel;
cl::Kernel        did_kernel;
cl::Kernel        dlower_kernel;
cl::Kernel        dupper_kernel;
cl::Kernel        dsub_kernel;
// cl::Kernel        slgauss_kernel;
// cl::Kernel        slbin_kernel;
// cl::Kernel        slpois_kernel; 
// cl::Kernel        sexp_kernel;
// cl::Kernel        dlgauss_kernel;
// cl::Kernel        dlbin_kernel;
// cl::Kernel        dlpois_kernel; 
cl::Kernel        mdlgauss_kernel;
cl::Kernel        mdlbin_kernel;
cl::Kernel        mdlpois_kernel; 
cl::Kernel        dllp_kernel;
cl::Kernel        dra_kernel;
cl::Kernel        drs_kernel; 
cl::Kernel        sra_kernel;
cl::Kernel        srs_kernel; 
cl::Kernel        dssa_kernel;
cl::Kernel        dsss_kernel; 
cl::Kernel        sssa_kernel;
cl::Kernel        ssss_kernel; 
cl::Kernel        test_kernel;
cl::Kernel        red_to_s_by_unit_kernel;
cl::Kernel        red_to_a_by_unit_kernel;
cl::Kernel        red_a_to_s_by_unit_kernel;
cl::Kernel        red_by_lv_kernel;
cl::Kernel        mlp_kernel;

void cl_dbuffer_output(cl::Buffer * buf, size_t rows, size_t cols,
		      size_t rows_out, size_t cols_out) {

  if (buf != nullptr) {
    size_t size = rows*cols;
    double* buffer = new double[size];
    
    ocl_q->enqueueReadBuffer(*buf, CL_TRUE, 0, size*sizeof(cl_double),
			     buffer);
    ocl_q->finish();
    
    std::cout.width(16);
    std::cout.precision(14);

    if (rows_out == 0) rows_out = rows;
    if (cols_out == 0) cols_out = cols;
    for(size_t i = 0; i< rows_out; i++) {
      for(size_t j = 0; j< cols_out; j++) {
	std::cout << buffer[i*cols + j] << " ";
      }
      std::cout << "\n";
    }
    cout << "\n";
    delete [] buffer;
  }
};
  