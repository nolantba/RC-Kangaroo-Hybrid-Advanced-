//Workaround for GCC 11+ AMX intrinsics incompatibility with CUDA
#ifndef CUDA_COMPAT_H
#define CUDA_COMPAT_H

// Block AMX tile intrinsics before they cause problems
#define _AMXTILEINTRIN_H_INCLUDED
#define _AMX_TILEINTRIN_H_INCLUDED

// Define dummy AMX functions if needed
#ifdef __cplusplus
extern "C" {
#endif

static inline void _tile_loadconfig(const void* __config) {}
static inline void _tile_storeconfig(void* __config) {}

#ifdef __cplusplus
}
#endif

#endif // CUDA_COMPAT_H
