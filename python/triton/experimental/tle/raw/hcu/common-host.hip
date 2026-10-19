#include <cstring>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#include <cstddef>
#include <cstdint>

namespace {
bool rocshmem_initialized = false;
}

extern "C" size_t rocshmem_unique_id_size() {
  return sizeof(rocshmem_uniqueid_t);
}

extern "C" int rocshmem_get_unique_id_bytes(void *uid_buffer,
                                             size_t uid_buffer_size) {
  if (uid_buffer == nullptr || uid_buffer_size < sizeof(rocshmem_uniqueid_t)) {
    return -1;
  }
  rocshmem_uniqueid_t uid;
  const int result = rocshmem_get_uniqueid(&uid);
  if (result != 0) {
    return result;
  }
  std::memcpy(uid_buffer, &uid, sizeof(uid));
  return 0;
}

extern "C" int rocshmem_init_from_torch_distributed(
    int rank, int nranks, int hip_device, void *uid_buffer,
    size_t uid_buffer_size) {
  if (rocshmem_initialized) {
    return 0;
  }
  if (uid_buffer == nullptr || uid_buffer_size < sizeof(rocshmem_uniqueid_t)) {
    return -1;
  }

  const hipError_t hip_result = hipSetDevice(hip_device);
  if (hip_result != hipSuccess) {
    return -1000 - static_cast<int>(hip_result);
  }

  rocshmem_uniqueid_t uid;
  std::memcpy(&uid, uid_buffer, sizeof(uid));

  rocshmem_init_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  int result = rocshmem_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
  if (result != 0) {
    return result;
  }
  result = rocshmem_init_attr(ROCSHMEM_INIT_WITH_UNIQUEID, &attr);
  if (result == 0) {
    rocshmem_initialized = true;
  }
  return result;
}

extern "C" int rocshmem_finalize_from_torch_distributed() {
  if (rocshmem_initialized) {
    rocshmem_finalize();
    rocshmem_initialized = false;
  }
  return 0;
}

extern "C" int tle_hip_stream_write_value64(void *stream_handle, void *ptr,
                                             uint64_t value) {
  return static_cast<int>(hipStreamWriteValue64(
      reinterpret_cast<hipStream_t>(stream_handle), ptr, value, 0));
}

extern "C" int tle_hip_memcpy_async(void *dst, const void *src, size_t size,
                                     void *stream_handle) {
  return static_cast<int>(hipMemcpyAsync(
      dst, src, size, hipMemcpyDefault,
      reinterpret_cast<hipStream_t>(stream_handle)));
}

extern "C" const char *tle_hip_error_string(int error) {
  return hipGetErrorString(static_cast<hipError_t>(error));
}

// ABI-compatible aliases: existing tutorial Python code can keep using the old
// exported symbol names while importing this rocSHMEM utility module.
extern "C" size_t nvshmem_unique_id_size() {
  return rocshmem_unique_id_size();
}

extern "C" int nvshmem_get_unique_id_bytes(void *uid_buffer,
                                            size_t uid_buffer_size) {
  return rocshmem_get_unique_id_bytes(uid_buffer, uid_buffer_size);
}

extern "C" int nvshmem_init_from_torch_distributed(
    int rank, int nranks, int hip_device, void *uid_buffer,
    size_t uid_buffer_size) {
  return rocshmem_init_from_torch_distributed(
      rank, nranks, hip_device, uid_buffer, uid_buffer_size);
}

extern "C" int nvshmem_finalize_from_torch_distributed() {
  return rocshmem_finalize_from_torch_distributed();
}

// #include <cstring>
// #include <cuda_runtime.h>
// #include <rocshmem.h>
// #include <rocshmemx.h>
// #include <stdint.h>
// #include <stdio.h>

// #undef CUDA_CHECK
// #define CUDA_CHECK(stmt)                                                       \
//   do {                                                                         \
//     cudaError_t result = (stmt);                                               \
//     if (cudaSuccess != result) {                                               \
//       fprintf(stderr, "[%s:%d] cuda failed with %s \n", __FILE__, __LINE__,    \
//               cudaGetErrorString(result));                                     \
//       exit(-1);                                                                \
//     }                                                                          \
//   } while (0)

// extern "C" int nvshmem_get_unique_id_bytes(void *uid_buffer,
//                                            size_t uid_buffer_size) {
//   if (uid_buffer_size < sizeof(nvshmemx_uniqueid_t)) {
//     return -1;
//   }

//   nvshmemx_uniqueid_t uid;
//   nvshmemx_get_uniqueid(&uid);
//   memcpy(uid_buffer, &uid, sizeof(uid));
//   return 0;
// }

// extern "C" int nvshmem_init_from_torch_distributed(int rank, int nranks,
//                                                    int cuda_device,
//                                                    void *uid_buffer,
//                                                    size_t uid_buffer_size) {
//   if (uid_buffer_size < sizeof(nvshmemx_uniqueid_t)) {
//     return -1;
//   }

//   CUDA_CHECK(cudaSetDevice(cuda_device));

//   nvshmemx_uniqueid_t uid;
//   memcpy(&uid, uid_buffer, sizeof(uid));

//   nvshmemx_init_attr_t attr;
//   memset(&attr, 0, sizeof(attr));
//   nvshmemx_set_attr_uniqueid_args(rank, nranks, &uid, &attr);
//   nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);

//   return 0;
// }

// extern "C" int nvshmem_finalize_from_torch_distributed() {
//   nvshmem_finalize();

//   return 0;
// }
