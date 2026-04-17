////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 HCU Research and HCU HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.hcu.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HCU_HSA_KERNEL_CODE_H
#define HCU_HSA_KERNEL_CODE_H

#include "hcu_hsa_common.h"
#include "hsa.h"

// HCU Kernel Code Version Enumeration Values.
typedef uint32_t hcu_kernel_code_version32_t;
enum hcu_kernel_code_version_t {
  HCU_KERNEL_CODE_VERSION_MAJOR = 1,
  HCU_KERNEL_CODE_VERSION_MINOR = 1
};

// HCU Machine Kind Enumeration Values.
typedef uint16_t hcu_machine_kind16_t;
enum hcu_machine_kind_t {
  HCU_MACHINE_KIND_UNDEFINED = 0,
  HCU_MACHINE_KIND_HCUGPU = 1
};

// HCU Machine Version.
typedef uint16_t hcu_machine_version16_t;

// HCU Float Round Mode Enumeration Values.
enum hcu_float_round_mode_t {
  HCU_FLOAT_ROUND_MODE_NEAREST_EVEN = 0,
  HCU_FLOAT_ROUND_MODE_PLUS_INFINITY = 1,
  HCU_FLOAT_ROUND_MODE_MINUS_INFINITY = 2,
  HCU_FLOAT_ROUND_MODE_ZERO = 3
};

// HCU Float Denorm Mode Enumeration Values.
enum hcu_float_denorm_mode_t {
  HCU_FLOAT_DENORM_MODE_FLUSH_SOURCE_OUTPUT = 0,
  HCU_FLOAT_DENORM_MODE_FLUSH_OUTPUT = 1,
  HCU_FLOAT_DENORM_MODE_FLUSH_SOURCE = 2,
  HCU_FLOAT_DENORM_MODE_NO_FLUSH = 3
};

// HCU Compute Program Resource Register One.
typedef uint32_t hcu_compute_pgm_rsrc_one32_t;
enum hcu_compute_pgm_rsrc_one_t {
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_GRANULATED_WORKITEM_VGPR_COUNT, 0, 6),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_GRANULATED_WAVEFRONT_SGPR_COUNT, 6, 4),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_PRIORITY, 10, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_32, 12, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_FLOAT_ROUND_MODE_16_64, 14, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_32, 16, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_FLOAT_DENORM_MODE_16_64, 18, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_PRIV, 20, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_ENABLE_DX10_CLAMP, 21, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_DEBUG_MODE, 22, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_ENABLE_IEEE_MODE, 23, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_BULKY, 24, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_CDBG_USER, 25, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_ONE_RESERVED1, 26, 6)
};

// HCU System VGPR Workitem ID Enumeration Values.
enum hcu_system_vgpr_workitem_id_t {
  HCU_SYSTEM_VGPR_WORKITEM_ID_X = 0,
  HCU_SYSTEM_VGPR_WORKITEM_ID_X_Y = 1,
  HCU_SYSTEM_VGPR_WORKITEM_ID_X_Y_Z = 2,
  HCU_SYSTEM_VGPR_WORKITEM_ID_UNDEFINED = 3
};

// HCU Compute Program Resource Register Two.
typedef uint32_t hcu_compute_pgm_rsrc_two32_t;
enum hcu_compute_pgm_rsrc_two_t {
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_PRIVATE_SEGMENT_WAVE_BYTE_OFFSET, 0, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_USER_SGPR_COUNT, 1, 5),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_TRAP_HANDLER, 6, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_X, 7, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Y, 8, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_ID_Z, 9, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_SGPR_WORKGROUP_INFO, 10, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_VGPR_WORKITEM_ID, 11, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_ADDRESS_WATCH, 13, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_MEMORY_VIOLATION, 14, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_GRANULATED_LDS_SIZE, 15, 9),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INVALID_OPERATION, 24, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_FP_DENORMAL_SOURCE, 25, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_DIVISION_BY_ZERO, 26, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_OVERFLOW, 27, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_UNDERFLOW, 28, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_IEEE_754_FP_INEXACT, 29, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_ENABLE_EXCEPTION_INT_DIVISION_BY_ZERO, 30, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_COMPUTE_PGM_RSRC_TWO_RESERVED1, 31, 1)
};

// HCU Element Byte Size Enumeration Values.
enum hcu_element_byte_size_t {
  HCU_ELEMENT_BYTE_SIZE_2 = 0,
  HCU_ELEMENT_BYTE_SIZE_4 = 1,
  HCU_ELEMENT_BYTE_SIZE_8 = 2,
  HCU_ELEMENT_BYTE_SIZE_16 = 3
};

// HCU Kernel Code Properties.
typedef uint32_t hcu_kernel_code_properties32_t;
enum hcu_kernel_code_properties_t {
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_QUEUE_PTR, 2, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 3, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_DISPATCH_ID, 4, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 6, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_X, 7, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Y, 8, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_SGPR_GRID_WORKGROUP_COUNT_Z, 9, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_WAVEFRONT_SIZE32, 10, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_RESERVED1, 11, 5),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_ENABLE_ORDERED_APPEND_GDS, 16, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_PRIVATE_ELEMENT_SIZE, 17, 2),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_IS_PTR64, 19, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK, 20, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_IS_DEBUG_ENABLED, 21, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_IS_XNACK_ENABLED, 22, 1),
  HCU_HSA_BITS_CREATE_ENUM_ENTRIES(HCU_KERNEL_CODE_PROPERTIES_RESERVED2, 23, 9)
};

// HCU Power Of Two Enumeration Values.
typedef uint8_t hcu_powertwo8_t;
enum hcu_powertwo_t {
  HCU_POWERTWO_1 = 0,
  HCU_POWERTWO_2 = 1,
  HCU_POWERTWO_4 = 2,
  HCU_POWERTWO_8 = 3,
  HCU_POWERTWO_16 = 4,
  HCU_POWERTWO_32 = 5,
  HCU_POWERTWO_64 = 6,
  HCU_POWERTWO_128 = 7,
  HCU_POWERTWO_256 = 8
};

// HCU Enabled Control Directive Enumeration Values.
typedef uint64_t hcu_enabled_control_directive64_t;
enum hcu_enabled_control_directive_t {
  HCU_ENABLED_CONTROL_DIRECTIVE_ENABLE_BREAK_EXCEPTIONS = 1,
  HCU_ENABLED_CONTROL_DIRECTIVE_ENABLE_DETECT_EXCEPTIONS = 2,
  HCU_ENABLED_CONTROL_DIRECTIVE_MAX_DYNAMIC_GROUP_SIZE = 4,
  HCU_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_GRID_SIZE = 8,
  HCU_ENABLED_CONTROL_DIRECTIVE_MAX_FLAT_WORKGROUP_SIZE = 16,
  HCU_ENABLED_CONTROL_DIRECTIVE_REQUIRED_DIM = 32,
  HCU_ENABLED_CONTROL_DIRECTIVE_REQUIRED_GRID_SIZE = 64,
  HCU_ENABLED_CONTROL_DIRECTIVE_REQUIRED_WORKGROUP_SIZE = 128,
  HCU_ENABLED_CONTROL_DIRECTIVE_REQUIRE_NO_PARTIAL_WORKGROUPS = 256
};

// HCU Exception Kind Enumeration Values.
typedef uint16_t hcu_exception_kind16_t;
enum hcu_exception_kind_t {
  HCU_EXCEPTION_KIND_INVALID_OPERATION = 1,
  HCU_EXCEPTION_KIND_DIVISION_BY_ZERO = 2,
  HCU_EXCEPTION_KIND_OVERFLOW = 4,
  HCU_EXCEPTION_KIND_UNDERFLOW = 8,
  HCU_EXCEPTION_KIND_INEXACT = 16
};

// HCU Control Directives.
#define HCU_CONTROL_DIRECTIVES_ALIGN_BYTES 64
#define HCU_CONTROL_DIRECTIVES_ALIGN __ALIGNED__(HCU_CONTROL_DIRECTIVES_ALIGN_BYTES)
typedef HCU_CONTROL_DIRECTIVES_ALIGN struct hcu_control_directives_s {
  hcu_enabled_control_directive64_t enabled_control_directives;
  uint16_t enable_break_exceptions;
  uint16_t enable_detect_exceptions;
  uint32_t max_dynamic_group_size;
  uint64_t max_flat_grid_size;
  uint32_t max_flat_workgroup_size;
  uint8_t required_dim;
  uint8_t reserved1[3];
  uint64_t required_grid_size[3];
  uint32_t required_workgroup_size[3];
  uint8_t reserved2[60];
} hcu_control_directives_t;

// HCU Kernel Code.
#define HCU_ISA_ALIGN_BYTES 256
#define HCU_KERNEL_CODE_ALIGN_BYTES 64
#define HCU_KERNEL_CODE_ALIGN __ALIGNED__(HCU_KERNEL_CODE_ALIGN_BYTES)
typedef HCU_KERNEL_CODE_ALIGN struct hcu_kernel_code_s {
  hcu_kernel_code_version32_t hcu_kernel_code_version_major;
  hcu_kernel_code_version32_t hcu_kernel_code_version_minor;
  hcu_machine_kind16_t hcu_machine_kind;
  hcu_machine_version16_t hcu_machine_version_major;
  hcu_machine_version16_t hcu_machine_version_minor;
  hcu_machine_version16_t hcu_machine_version_stepping;
  int64_t kernel_code_entry_byte_offset;
  int64_t kernel_code_prefetch_byte_offset;
  uint64_t kernel_code_prefetch_byte_size;
  uint64_t max_scratch_backing_memory_byte_size;
  hcu_compute_pgm_rsrc_one32_t compute_pgm_rsrc1;
  hcu_compute_pgm_rsrc_two32_t compute_pgm_rsrc2;
  hcu_kernel_code_properties32_t kernel_code_properties;
  uint32_t workitem_private_segment_byte_size;
  uint32_t workgroup_group_segment_byte_size;
  uint32_t gds_segment_byte_size;
  uint64_t kernarg_segment_byte_size;
  uint32_t workgroup_fbarrier_count;
  uint16_t wavefront_sgpr_count;
  uint16_t workitem_vgpr_count;
  uint16_t reserved_vgpr_first;
  uint16_t reserved_vgpr_count;
  uint16_t reserved_sgpr_first;
  uint16_t reserved_sgpr_count;
  uint16_t debug_wavefront_private_segment_offset_sgpr;
  uint16_t debug_private_segment_buffer_sgpr;
  hcu_powertwo8_t kernarg_segment_alignment;
  hcu_powertwo8_t group_segment_alignment;
  hcu_powertwo8_t private_segment_alignment;
  hcu_powertwo8_t wavefront_size;
  int32_t call_convention;
  uint8_t reserved1[12];
  uint64_t runtime_loader_kernel_symbol;
  hcu_control_directives_t control_directives;
} hcu_kernel_code_t;

// TODO: this struct should be completely gone once debugger designs/implements
// Debugger APIs.
typedef struct hcu_runtime_loader_debug_info_s {
  const void* elf_raw;
  size_t elf_size;
  const char *kernel_name;
  const void *owning_segment;
} hcu_runtime_loader_debug_info_t;

#endif // HCU_HSA_KERNEL_CODE_H
