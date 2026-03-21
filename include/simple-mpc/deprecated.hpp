#pragma once

#if defined(__has_cpp_attribute)
#  if __has_cpp_attribute(deprecated)
#    define SIMPLE_MPC_DEPRECATED [[deprecated]]
#  else
#    define SIMPLE_MPC_DEPRECATED
#  endif
#else
#  define SIMPLE_MPC_DEPRECATED
#endif
