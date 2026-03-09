#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <cstring>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    // b mean batch..

    // first Assume always matrix is divide by batch size
    for(size_t b = 0; b < m; b += batch){
        size_t cur_batch = (m - b < batch) ? (m - b) : batch;
        float *logits = new float[cur_batch * k]; // for logtis
        float *Iy = new float[cur_batch * k]; // for one hot vector
        float *Z = new float[cur_batch * k]; // for
        // float *grad = new float[n * k];
        std::memset(logits, 0, cur_batch * k * sizeof(float));
        std::memset(Iy, 0, cur_batch * k * sizeof(float));
        std::memset(Z, 0, cur_batch * k * sizeof(float));
        // std::memset(grad, 0, n * k * sizeof(float));

        // 1. get logits X @ theta
        // this outer loop for row
        for(size_t row = 0; row < cur_batch; row++){
            float row_sum = 0.0f;
            // this col loop for is for k class for logits and theta
            for(size_t col = 0; col < k; col++){
                float sum = 0.0f;
                // this loop is for n dimesion when X @ theta
                for(size_t j = 0; j < n; j++){
                    sum += X[b * n + row * n + j] * theta[j * k + col];
                }
                logits[row * k + col] = sum; // Store final result + exp first for softmax
                Z[row * k + col] = std::exp(sum);
                row_sum += Z[row * k + col];
            }
            // 2. get one hot vector of y
            unsigned char label = y[b + row];
            Iy[row * k + label] = 1.0f;
            // 3. get Z from normalize(exp(logits))
            for(size_t col = 0; col < k; col++){
                Z[row * k + col] /= row_sum;
                Z[row * k + col] -= Iy[row * k + col];
            }
        }
        // 4. get gradient and theta 
        // gradient -> X_batch @ (Z - Iy)
        

        // for(size_t row = 0; row < n; row++){
        //     for(size_t col = 0; col < k; col++ ){
        //         float sum = 0.0f;
        //         for(size_t j = 0; j < cur_batch; j++){
        //             // both are column fast so if change row first maybe fast...
        //             sum += X[(b +  j) * n + row] * Z[j * k + col];
        //         }
        //         theta[row * k + col] -= lr * sum / cur_batch;
        //     }
        // } 
        // --- 2. Gradient Update (Optimized) ---
        // New Order: Batch (i) -> Input Feature (j) -> Class (col)
        // This ensures we read X sequentially!

        // Pre-calculate the learning step to avoid division inside loops
        float step = lr / static_cast<float>(cur_batch);

        for (size_t i = 0; i < cur_batch; i++) {     // 1. Iterate over Batch (Sequential X access)
            
            for (size_t j = 0; j < n; j++) {         // 2. Iterate over Input Features (Rows of Theta)
                
                // We read X[(b+i)*n + j]. 
                // Since 'i' and 'j' increment sequentially, this is perfectly 
                // contiguous memory access. CPU prefetching works great here.
                float x_val = X[(b + i) * n + j];

                for (size_t col = 0; col < k; col++) { // 3. Iterate over Classes (Cols of Theta)
                    
                    // Update Theta In-Place
                    // theta[j*k + col]: Sequential access
                    // Z[i*k + col]:     Sequential access
                    
                    theta[j * k + col] -= step * (x_val * Z[i * k + col]);
                }
            }
        }

        
        delete[] logits;
        delete[] Iy;
        delete[] Z;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
