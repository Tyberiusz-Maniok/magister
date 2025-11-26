#ifndef CUDA_NVHPC_COMPAT_H
#define CUDA_NVHPC_COMPAT_H

// Workaround for CUDA 12.6 + NVHPC 24.9 header incompatibility
// Include driver_types.h BEFORE cuda_runtime_api.h to ensure
// cudaAsyncCallback and related types are defined
#include <driver_types.h>

#endif // CUDA_NVHPC_COMPAT_H

