#include <cuda_runtime_api.h>

namespace groundeddino_vl {
int get_cudart_version() {
  return CUDART_VERSION;
}
} // namespace groundeddino_vl
