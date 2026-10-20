#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

extern "C" __device__ __attribute__((always_inline)) void
simple_shift(__attribute__((address_space(1))) int *destination) {
  // rocSHMEM APIs without _wave/_wg suffixes are single-thread APIs.
  if (threadIdx.x == 0) {
    const int mype = rocshmem_my_pe();
    const int npes = rocshmem_n_pes();
    const int peer = (mype + 1) % npes;
    rocshmem_int_p((int *)destination, mype, peer);
  }
}
