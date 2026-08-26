// Modified copy_matrix function and helper function 

#include <cstdint>
#include <cstring>

static std::uint64_t hash_matrix(const gsl_matrix* matrix)
{
    /*
        Function generates a unique hash from the matrix structure 
    */

   /// intial value
    std::uint64_t hash = 1469598103934665603ULL;

    auto mix = [&hash](std::uint64_t value) {
        hash ^= value;
        hash *= 1099511628211ULL;
    };

    mix(static_cast<std::uint64_t>(matrix->size1));
    mix(static_cast<std::uint64_t>(matrix->size2));

    for (size_t row = 0; row < matrix->size1; row++) {
        const double* row_data =
            matrix->data + row * matrix->tda;

        for (size_t col = 0; col < matrix->size2; col++) {
            std::uint64_t bits = 0;

            static_assert(
                sizeof(bits) == sizeof(row_data[col]),
                "Unexpected double size"
            );

            std::memcpy(
                &bits,
                &row_data[col],
                sizeof(bits)
            );

            mix(bits);
        }
    }

    return hash;
}


int copy_matrix_to_ocl(
    const gsl_matrix* src,
    cl::Buffer*& dest,
    bool values,
    cl_mem_flags flags
) {

    /*
        Function transfers structures to the GPU only when the equivalent CPU structure has been modified.  
    */

    static std::unordered_map<cl::Buffer*, std::uint64_t>
        previous_hashes;

    if (src == nullptr) {
        std::cerr << "copy_matrix_to_ocl received a null source matrix"
                  << std::endl;
        return CL_INVALID_VALUE;
    }

    const size_t element_count = src->size1 * src->size2;
    const size_t byte_size = element_count * sizeof(cl_double);

    if (byte_size == 0) {
        std::cerr << "Cannot create a zero-sized OpenCL buffer"
                  << std::endl;
        return CL_INVALID_BUFFER_SIZE;
    }

    cl_int err = CL_SUCCESS;
    cl_int info_error = CL_SUCCESS;
    bool buffer_created = false;

    if (dest != nullptr) {
        const size_t existing_size =
            dest->getInfo<CL_MEM_SIZE>(&info_error);

        if (info_error != CL_SUCCESS ||
            existing_size != byte_size) {

            // Remove the checksum associated with the old buffer.
            previous_hashes.erase(dest);

            delete dest;
            dest = nullptr;
        }
    }

    if (dest == nullptr) {
        dest = new cl::Buffer(
            *ocl_ctx,
            flags,
            byte_size,
            nullptr,
            &err
        );

        if (err != CL_SUCCESS) {
            std::cerr
                << "Creating OpenCL buffer failed with error code "
                << err
                << std::endl;

            delete dest;
            dest = nullptr;
            return err;
        }

        buffer_created = true;
    }

    const std::uint64_t current_hash = hash_matrix(src);
    const auto previous = previous_hashes.find(dest);

    const bool changed =
        buffer_created ||
        previous == previous_hashes.end() ||
        current_hash != previous->second;

    if (values && changed) {

        if (src->tda == src->size2) {
            err = ocl_q->enqueueWriteBuffer(
                *dest,
                CL_TRUE,
                0,
                byte_size,
                src->data
            );
        }
        else {
      
            std::vector<cl_double> packed(element_count);

            for (size_t row = 0; row < src->size1; row++) {
                const double* source_row =
                    src->data + row * src->tda;

                std::copy(
                    source_row,
                    source_row + src->size2,
                    packed.begin() + row * src->size2
                );
            }

            err = ocl_q->enqueueWriteBuffer(
                *dest,
                CL_TRUE,
                0,
                byte_size,
                packed.data()
            );
        }

        if (err != CL_SUCCESS) {
            std::cerr
                << "Copying matrix to OpenCL failed with error code "
                << err
                << std::endl;

        }
        else {
            previous_hashes[dest] = current_hash;
            changed_count++;
        }
    }
    else if (values) {
        unchanged_count++;
    }

    return err;
}



