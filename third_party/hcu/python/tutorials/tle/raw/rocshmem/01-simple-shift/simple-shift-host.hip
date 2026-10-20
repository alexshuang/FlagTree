#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

namespace {

int hip_status(hipError_t result, const char *operation) {
  if (result == hipSuccess) {
    return 0;
  }
  std::fprintf(stderr, "%s failed: %s\n", operation, hipGetErrorString(result));
  return -1000 - static_cast<int>(result);
}

} // namespace

extern "C" int simple_shift_before_launch(
    int *mype, int *npes, int *mype_in_node, int *npes_in_node,
    hipStream_t *stream, int **dst, int **data_h) {
  if (mype == nullptr || npes == nullptr || mype_in_node == nullptr ||
      npes_in_node == nullptr || stream == nullptr || dst == nullptr ||
      data_h == nullptr) {
    return -1;
  }

  *mype = rocshmem_my_pe();
  *npes = rocshmem_n_pes();
  *mype_in_node = rocshmem_team_my_pe(ROCSHMEM_TEAM_SHARED);
  *npes_in_node = rocshmem_team_n_pes(ROCSHMEM_TEAM_SHARED);

  // init_rocshmem_by_torch_pg() has already selected the PE's HIP device.
  int result = hip_status(hipStreamCreate(stream), "hipStreamCreate");
  if (result != 0) {
    return result;
  }

  *dst = static_cast<int *>(rocshmem_malloc(sizeof(int)));
  if (*dst == nullptr) {
    hipStreamDestroy(*stream);
    *stream = nullptr;
    return -2;
  }

  *data_h = static_cast<int *>(std::malloc(sizeof(int)));
  if (*data_h == nullptr) {
    rocshmem_free(*dst);
    *dst = nullptr;
    hipStreamDestroy(*stream);
    *stream = nullptr;
    return -3;
  }

  result = hip_status(hipMemsetAsync(*dst, 0xff, sizeof(int), *stream),
                      "hipMemsetAsync");
  if (result == 0) {
    result = hip_status(hipStreamSynchronize(*stream), "hipStreamSynchronize");
  }
  if (result != 0) {
    std::free(*data_h);
    *data_h = nullptr;
    rocshmem_free(*dst);
    *dst = nullptr;
    hipStreamDestroy(*stream);
    *stream = nullptr;
  }
  return result;
}

extern "C" int simple_shift_after_launch(hipStream_t stream, void *dst,
                                          void *data_h, int mype, int npes) {
  if (stream == nullptr || dst == nullptr || data_h == nullptr) {
    return -1;
  }

  // This stream-ordered barrier waits for every PE and completes prior remote
  // updates before the local result is copied to host memory.
  rocshmem_barrier_all_on_stream(stream);
  int result = hip_status(
      hipMemcpyAsync(data_h, dst, sizeof(int), hipMemcpyDeviceToHost, stream),
      "hipMemcpyAsync");
  if (result == 0) {
    result = hip_status(hipStreamSynchronize(stream), "hipStreamSynchronize");
  }

  if (result == 0) {
    const int received = *static_cast<int *>(data_h);
    const int expected = (mype - 1 + npes) % npes;
    std::printf("PE %d/%d: received message %d (expected %d)\n", mype, npes,
                received, expected);
    if (received != expected) {
      result = -4;
    }
  }

  // rocshmem_free is collective, so all PEs must reach this point.
  rocshmem_free(dst);
  std::free(data_h);
  const int destroy_result =
      hip_status(hipStreamDestroy(stream), "hipStreamDestroy");
  return result != 0 ? result : destroy_result;
}
