# basalt: a local inference engine for one NVIDIA GPU.
cmake_minimum_required(VERSION 3.24)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "CUDA architectures" FORCE)
endif()
project(basalt VERSION 0.5.6 LANGUAGES C CXX CUDA)
add_compile_definitions(BASALT_VERSION="${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(BUILD_TESTING "Build the tests and parity programs" OFF)
# Covers the project's own C++ and CUDA sources only; off by default so a newer compiler's new warning cannot break a user build.
option(WERROR "Treat compiler warnings as errors" OFF)
# Optional peer architectures: building the peer-tier kernels for sm_89 (Ada) in addition to sm_120
# allows a non-Blackwell card to serve as the expert tier, but adds build time.
option(PEER_ADA "Compile peer kernels for sm_89 (Ada) in addition to sm_120" OFF)
if(BUILD_TESTING)
  enable_testing()
endif()

# Every test is registered through one of these so it carries a label: `cpu` tests run with no GPU or driver present,
# `gpu` tests need a device. `ctest -L cpu` is what the hosted CI runs.
function(_basalt_labelled_test label)
  add_test(${ARGN})
  cmake_parse_arguments(_t "" "NAME" "" ${ARGN})
  set_property(TEST ${_t_NAME} APPEND PROPERTY LABELS ${label})
endfunction()
function(basalt_cpu_test)
  _basalt_labelled_test(cpu ${ARGN})
endfunction()
function(basalt_gpu_test)
  _basalt_labelled_test(gpu ${ARGN})
endfunction()

# the CUDA language gives the .cu compile rules but not the imported targets, so `CUDA::cudart` is brought in
# explicitly for the host-side programs that call the runtime themselves
find_package(CUDAToolkit REQUIRED)
# Main kernels (NVFP4 tensor cores, TMA, Blackwell instructions) compile for sm_120 only.
# When PEER_ADA is enabled, peer kernels (Q8_1/Q2_0 expert rows, standard CUDA) additionally compile
# for sm_89 (Ada) so a non-Blackwell card can serve as the expert tier.
foreach(_a IN LISTS CMAKE_CUDA_ARCHITECTURES)
  string(REGEX REPLACE "[-a].*$" "" _base "${_a}")     # 120-real / 120a are still 120
  if(NOT _base STREQUAL "120")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES is '${CMAKE_CUDA_ARCHITECTURES}': basalt builds for compute capability 12.0 only (peer-arch kernels are compiled separately below)")
  endif()
endforeach()
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(FATAL_ERROR "CUDA ${CMAKE_CUDA_COMPILER_VERSION}: basalt needs CUDA 13.0 or newer")
endif()
message(STATUS "basalt: CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
set(_gpu_runtime_target CUDA::cudart)
set(_gpu_blas_target CUDA::cublas)
set(_gpu_include_directories ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# MSVC reports __cplusplus as 199711L unless /Zc:__cplusplus is given. cuda.h applies alignas(TENSOR_MAP_ALIGN) to
# CUtensorMap only when __cplusplus says C++11 or later, so without the option the type keeps the 8-byte alignment
# of its single array member and the TMA descriptors inside a __grid_constant__ kernel-argument struct fault on a
# misaligned address. The option has to be global: the alignment is decided in the translation unit that reads the
# header, which is not necessarily one that links the target carrying a compile option.
if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:/Zc:__cplusplus>)
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>)
  # nvcc takes host-compiler options through -Xcompiler, and this reaches every CUDA target.
  string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:__cplusplus")
endif()
# CCCL in CUDA 13.2 and later rejects cl.exe's traditional preprocessor (C1189); nvcc's default Windows host is cl.exe.
if(WIN32)
  file(TO_CMAKE_PATH "${CMAKE_CUDA_HOST_COMPILER}" _cuda_host)
  get_filename_component(_cuda_host "${_cuda_host}" NAME)
  string(TOLOWER "${_cuda_host}" _cuda_host)
  if(_cuda_host STREQUAL "" OR _cuda_host MATCHES "^cl(\\.exe)?$")
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=/Zc:preprocessor")
  endif()
endif()

# The `warnings` and `platform` targets, one file shared with the standalone `tools/vision` configuration.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/platform.cmake)

# BASALT_MODELS: which models this build contains (all of them by default).
include(${CMAKE_CURRENT_LIST_DIR}/cmake/models.cmake)

# The application manifest, for every program that names a file path in its command line. Windows decodes argv and
# the CRT's narrow opens in the machine's ANSI code page unless the image says otherwise, so the same pack path
# opens at one call site and fails at another. The manifest says the process code page is UTF-8 and that long paths
# are in use; `link.exe` merges the file into the RT_MANIFEST resource it already writes, so there is no mt.exe step
# and no external .manifest beside the exe. Nothing changes on Linux, where a path is bytes.
set(_basalt_manifest ${CMAKE_CURRENT_SOURCE_DIR}/src/platform/basalt.exe.manifest)
function(basalt_embed_manifest target)
  if(MSVC)
    target_link_options(${target} PRIVATE "/MANIFEST:EMBED" "/MANIFESTINPUT:${_basalt_manifest}")
    # a manifest edit has to relink the program, and the link command is the only thing that reads the file
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${_basalt_manifest})
  endif()
endfunction()

# The vector-extension flags per file, in each compiler's spelling. MSVC has no per-feature ISA switches: /arch:
# turns on the whole rung (and a TU built with it may use those instructions anywhere in itself, exactly as the
# GNU flags do), so the run-time probe still decides which of these objects runs.
#
# /arch:AVX512 predefines __AVX__, __AVX2__ and __AVX512F/BW/VL/DQ__ but leaves __F16C__, __AVX512VNNI__ and
# __AVX512VBMI__ undefined, while accepting the intrinsics of all three. The sources branch on those macros (the
# ggml dot products pick their VNNI form by __AVX512VNNI__), so they are defined to match what the GNU flags give:
# without it a file compiles down its no-VNNI path and runs slower than the same build on Linux. __BMI2__ is left
# alone because -mavx2 and -mavx512* do not enable BMI2 either, so that branch is out on both platforms.
if(MSVC)
  set(_isa_avx512_flags "/arch:AVX512;-D__F16C__=1;-D__AVX512VNNI__=1;-D__AVX512VBMI__=1")
  set(_isa_avx2_flags "/arch:AVX2;-D__F16C__=1")
  set(_isa_avxvnni_flags "/arch:AVX2;-D__F16C__=1")
  set(_isa_avxvnniint8_flags "/arch:AVX2;-D__F16C__=1")
  set(_isa_avx1_flags "/arch:AVX")
else()
  set(_isa_avx512_flags "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512vnni;-mavx512vbmi;-mf16c;-mfma")
  set(_isa_avx2_flags "-mavx2;-mfma;-mf16c")
  set(_isa_avxvnni_flags "-mavx2;-mfma;-mf16c;-mavxvnni;-mbmi2")
  # -mno-avx512f keeps the file below EVEX under a global -march=native: there vpdpbssd lands on ymm16+, which
  # only its AVX10.2 form encodes, so the assembler refuses it or emits an instruction AVX-512 CPUs do not run.
  set(_isa_avxvnniint8_flags "-mavx2;-mfma;-mf16c;-mavxvnniint8;-mno-avx512f")
  set(_isa_avx1_flags "-mavx")
endif()

# ---- artifact -------------------------------------------------------
# Header-only, and that is a deliberate property rather than an unfinished split: the GGUF reader and the
# scalar dequantizers hold no state worth hiding, and a header-only boundary cannot introduce a
# duplicate-symbol or missing-declaration bug into code that is already validated against the real artifact.
add_library(artifact INTERFACE)
target_include_directories(artifact INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(artifact INTERFACE platform warnings)

# ---- model adapters ---------------------------------------------------
# `model` is the registry and the adapter interface (shared code may use it). Each model is its own library under
# src/models/<arch>/; `models` is the composition root that registers them. tools/check_model_layering.sh keeps
# shared code from including src/models and models from including each other.
add_library(model STATIC src/model/registry.cpp src/model/rope_flags.cpp src/model/route_flags.cpp src/model/start.cpp src/model/batch_check.cpp)
target_compile_definitions(model PRIVATE BASALT_KNOWN_MODELS="${_basalt_known}" BASALT_BUILT_MODELS="${_basalt_built}")
target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(model PUBLIC artifact warnings)

if(BASALT_MODEL_HY4FLASH)
add_library(hy4flash_kernels STATIC src/models/hy4flash/kernels/mla.cu src/models/hy4flash/kernels/ops.cu
  src/models/hy4flash/kernels/indexer.cu)
target_include_directories(hy4flash_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hy4flash_kernels PUBLIC ${_gpu_runtime_target})
add_library(model_hy4flash STATIC src/models/hy4flash/adapter.cpp src/models/hy4flash/pretok_hy4.cpp src/models/hy4flash/pack.cpp
  src/models/hy4flash/model.cpp src/models/hy4flash/session.cpp
  src/models/hy4flash/vision.cpp src/models/hy4flash/vision_encoder.cpp)
target_include_directories(model_hy4flash PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src/program/tokenizer)
target_link_libraries(model_hy4flash PUBLIC model tokenizer hy4flash_kernels expert_tier tiers mmq peer_share kernels core vision_cuda ${_gpu_runtime_target})
# the DFly2 drafter (src/models/hy4flash/dfly*)
add_library(hy4flash_dfly_kernels STATIC src/models/hy4flash/dfly_kernels.cu)
target_include_directories(hy4flash_dfly_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(hy4flash_dfly_kernels PUBLIC ${_gpu_runtime_target})
target_sources(model_hy4flash PRIVATE src/models/hy4flash/dfly.cpp src/models/hy4flash/dfly_meta.cpp)
target_link_libraries(model_hy4flash PUBLIC hy4flash_dfly_kernels)
if(BUILD_TESTING)
  add_executable(hy4flash_dfly_test src/models/hy4flash/dfly_test.cpp src/models/hy4flash/dfly_meta.cpp)
  target_include_directories(hy4flash_dfly_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(hy4flash_dfly_test PRIVATE core)
  basalt_cpu_test(NAME hy4flash_dfly_test COMMAND hy4flash_dfly_test)
  add_executable(hy4flash_dfly_parity src/models/hy4flash/dfly_parity.cpp)
  target_include_directories(hy4flash_dfly_parity PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(hy4flash_dfly_parity PRIVATE model_hy4flash hy4flash_dfly_kernels hy4flash_kernels kernels core ${_gpu_runtime_target})
  basalt_gpu_test(NAME hy4flash_dfly_parity COMMAND hy4flash_dfly_parity)
  add_executable(hy4flash_dfly_real src/models/hy4flash/dfly_real.cpp)
  target_include_directories(hy4flash_dfly_real PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(hy4flash_dfly_real PRIVATE model_hy4flash hy4flash_dfly_kernels hy4flash_kernels kernels core ${_gpu_runtime_target})
endif()
if(BUILD_TESTING)
  add_executable(hy4flash_kernels_parity src/models/hy4flash/kernels/kernels_parity.cpp)
  target_include_directories(hy4flash_kernels_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(hy4flash_kernels_parity PRIVATE hy4flash_kernels mmq kernels ggml_quants ${_gpu_runtime_target})
  basalt_gpu_test(NAME hy4flash_kernels_parity COMMAND hy4flash_kernels_parity)
  add_executable(hy4flash_context_policy_test src/models/hy4flash/context_policy_test.cpp)
  target_include_directories(hy4flash_context_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/include)
  basalt_cpu_test(NAME hy4flash_context_policy_test COMMAND hy4flash_context_policy_test)
  add_executable(hy4flash_vision_test src/models/hy4flash/vision_test.cpp src/models/hy4flash/vision.cpp)
  target_include_directories(hy4flash_vision_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(hy4flash_vision_test PRIVATE vision_core warnings)
  basalt_gpu_test(NAME hy4_encoder_parity COMMAND hy4_encoder_parity)
  basalt_cpu_test(NAME hy4flash_vision_test COMMAND hy4flash_vision_test ${CMAKE_CURRENT_SOURCE_DIR}/assets/basalt-lockup-light.png)
  add_executable(hy4flash_pretok_test src/models/hy4flash/pretok_test.cpp src/models/hy4flash/pretok_hy4.cpp)
  target_include_directories(hy4flash_pretok_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/src/program/tokenizer)
  target_link_libraries(hy4flash_pretok_test PRIVATE tokenizer)
  basalt_cpu_test(NAME hy4flash_pretok_test COMMAND hy4flash_pretok_test)
endif()
endif()

# The composition root links the built models and sees their ids (BASALT_MODEL_<ARCH>) and the registration list.
add_library(models STATIC src/models/builtin.cpp)
target_include_directories(models PRIVATE ${CMAKE_BINARY_DIR}/generated)
foreach(_m IN LISTS BASALT_MODELS)
  string(TOUPPER ${_m} _up)
  target_link_libraries(models PUBLIC model_${_m})
  target_compile_definitions(models PUBLIC BASALT_MODEL_${_up}=1)
endforeach()

add_executable(basalt-gguf src/artifact/gguf_reader.cpp)
target_link_libraries(basalt-gguf PRIVATE artifact models)
basalt_embed_manifest(basalt-gguf)

add_executable(basalt-dequant src/artifact/dequant.cpp)
target_link_libraries(basalt-dequant PRIVATE artifact)
basalt_embed_manifest(basalt-dequant)
# this file is now nothing but the CLI entry point, and its inherited guard spells that as a "selftest"
target_compile_definitions(basalt-dequant PRIVATE DEQUANT_SELFTEST)
if(BUILD_TESTING)
  # the reader refuses a duplicate tensor name at open, naming it (a synthetic GGUF: no model, no GPU)
  add_executable(gguf_reader_test tests/artifact/gguf_reader_test.cpp)
  target_link_libraries(gguf_reader_test PRIVATE artifact)
  basalt_embed_manifest(gguf_reader_test)
  basalt_cpu_test(NAME gguf_reader_test COMMAND gguf_reader_test)
  # a split model's shards and their joint view: missing shard, split keys, duplicates, truncation (synthetic)
  add_executable(gguf_split_test tests/core/gguf_split_test.cpp)
  target_include_directories(gguf_split_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(gguf_split_test PRIVATE artifact model)
  basalt_embed_manifest(gguf_split_test)
  basalt_cpu_test(NAME gguf_split_test COMMAND gguf_split_test)
  # the adapter registry: lookup by architecture, geometry from metadata, the DeepSeek skeleton (synthetic)
  add_executable(adapter_registry_test tests/model/adapter_registry_test.cpp)
  target_include_directories(adapter_registry_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${_gpu_include_directories})
  target_link_libraries(adapter_registry_test PRIVATE model)
  basalt_embed_manifest(adapter_registry_test)
  basalt_cpu_test(NAME adapter_registry_test COMMAND adapter_registry_test)
  # several sessions' windows as one window of segments, and the composition graphs' eviction (header-only, CPU)
  add_executable(batch_window_test tests/model/batch_window_test.cpp)
  target_include_directories(batch_window_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${_gpu_include_directories})
  target_link_libraries(batch_window_test PRIVATE warnings)
  basalt_cpu_test(NAME batch_window_test COMMAND batch_window_test)
  # the real adapters, through the program's registration (needs the CUDA runtime, not a GPU's work)
  add_executable(adapter_builtin_test tests/model/adapter_builtin_test.cpp)
  target_link_libraries(adapter_builtin_test PRIVATE models)
  basalt_embed_manifest(adapter_builtin_test)
  basalt_gpu_test(NAME adapter_builtin_test COMMAND adapter_builtin_test)
  # the split of a layer's RAM experts between the cards (no GPU)
  add_executable(peer_take_test tests/prefill/peer_take_test.cpp)
  target_include_directories(peer_take_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  basalt_embed_manifest(peer_take_test)
  basalt_cpu_test(NAME peer_take_test COMMAND peer_take_test)
  # shared code must not include a model; models must not include each other
  # a check of the tree's text, in bash and python3: Linux's run covers it for every platform
  if(NOT WIN32)
    add_test(NAME model_layering COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_model_layering.sh)
    set_property(TEST model_layering APPEND PROPERTY LABELS cpu)
  endif()
  # the logit_bias protocol key: what parses and what is refused (host only)
  add_executable(logit_bias_test tests/core/logit_bias_test.cpp)
  target_include_directories(logit_bias_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  basalt_cpu_test(NAME logit_bias_test COMMAND logit_bias_test)
endif()

# ---- platform (host OS boundary) ------------------------------------
# Defined in cmake/platform.cmake, included above with `warnings`.

# ---- core (GPU runtime) ---------------------------------------------
add_library(core STATIC src/core/device.cu src/core/pinned.cu src/core/ram_load.cpp src/core/vmm.cpp src/core/elastic_array.cpp src/core/file_reads.cpp src/core/graph.cpp src/core/weights.cpp src/core/loader.cpp
                             src/core/host_plan.cpp src/core/checkpoint_transfer.cpp src/core/staged_copy.cpp src/core/prompt_stage.cpp)
target_include_directories(core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# The selected runtime is PUBLIC because graph.cpp is a HOST translation unit that calls cudaGraphLaunch and
# cudaEventQuery: the .cu sources get cudart through the CUDA language rules, a .cpp does not, and the
# linker error ("unresolved external symbol cudaGraphInstantiate") names the function rather than the cause.
target_link_libraries(core PUBLIC model_file artifact platform warnings ${_gpu_runtime_target})

add_executable(basalt-device src/program/device_main.cpp)
target_link_libraries(basalt-device PRIVATE core)

# Peer-tier compilation: the peer prefill mmq/iota kernels and the small utility kernels that run on the peer card additionally compile for sm_89 (Ada), so a non-Blackwell
# card can serve as the expert tier. Must be set BEFORE the targets that compile these files are defined.
if(PEER_ADA)
  add_compile_definitions(BASALT_PEER_ADA=1)
  set_source_files_properties(
    src/prefill/moe_mmq.cu
    src/prefill/mmq/mmq_host.cu
    src/prefill/mmq/quantize.cu
    src/prefill/mmq/instances/mmq_q2_0.cu
    src/prefill/mmq/instances/mmq_iq1_s.cu
    src/prefill/mmq/instances/mmq_iq1_m.cu
    src/prefill/mmq/instances/mmq_iq2_xxs.cu
    src/prefill/mmq/instances/mmq_iq2_xs.cu
    src/prefill/mmq/instances/mmq_iq2_s.cu
    src/prefill/mmq/instances/mmq_iq3_xxs.cu
    src/prefill/mmq/instances/mmq_iq3_s.cu
    src/prefill/mmq/instances/mmq_iq4_nl.cu
    src/prefill/mmq/instances/mmq_iq4_xs.cu
    src/prefill/mmq/instances/mmq_q8_0.cu
    src/prefill/mmq/instances/mmq_q5_0.cu
    src/prefill/mmq/instances/mmq_q5_1.cu
    src/prefill/mmq/instances/mmq_q2_k.cu
    src/prefill/mmq/instances/mmq_q3_k.cu
    src/prefill/mmq/instances/mmq_q4_k.cu
    src/prefill/mmq/instances/mmq_q5_k.cu
    src/prefill/mmq/instances/mmq_q6_k.cu
    src/kernels/cuda/iq_kernels.cu
    src/kernels/cuda/quantize_act.cu
    src/kernels/cuda/wait_abort.cu
    src/kernels/cuda/sampler.cu
    src/kernels/cuda/elementwise.cu
    PROPERTIES COMPILE_OPTIONS "--generate-code=arch=compute_89,code=sm_89")
endif()

# ---- kernels ----
# Naive first, each with a parity test against a reference that was itself validated against ggml.  The
# kernels link `artifact` for the scalar dequantizers, because using the VALIDATED scalar path as the
# reference is the whole point - a kernel checked against another copy of itself proves nothing.
add_library(kernels STATIC src/kernels/cuda/quantize_act.cu
                        src/kernels/cuda/sampler.cu
                        src/kernels/cuda/logprobs.cu
                        src/kernels/cuda/elementwise.cu
                        src/kernels/cuda/bf16_gemv.cu
                        src/kernels/cuda/native_mmvq.cu
                        src/kernels/cuda/q80_screen.cu
                        src/kernels/cuda/native_bf16.cu
                        src/kernels/cuda/dequant_bf16.cu
                        src/kernels/cuda/wait_abort.cu
                        src/kernels/cuda/iq_kernels.cu
                        src/kernels/cuda/gdn_window.cu
                        src/kernels/cuda/gdn_commit.cu
                        src/kernels/cuda/gdn_prompt.cu
                        src/kernels/cuda/dense_ops.cu
                        src/kernels/cuda/gqa_attn.cu
                        src/kernels/cuda/nvfp4_planes.cu
                        src/kernels/cuda/mhc.cu
                        src/kernels/cuda/mhc_plain.cu
                        src/kernels/cuda/router.cu
                        src/kernels/cuda/route_tail_skip.cu
                        src/kernels/cuda/row_norm.cu
                        src/ngram/ple_reader.cpp)
target_link_libraries(kernels PUBLIC core artifact)
# Match the pinned llama.cpp CUDA arithmetic contract for this adapter only.
set_source_files_properties(src/kernels/cuda/native_mmvq.cu src/kernels/cuda/native_bf16.cu
    src/kernels/cuda/q80_screen.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math")

# ---- engine -----------------------------------------------------
# The COMPOSED layer lives here and not in `core`, because it calls kernels and `kernels`
# already links `core` - putting `gdn_layer` in core would make the two mutually dependent.  A third
# library is the honest shape of the dependency graph: core knows about devices and weights, kernels know
# about arithmetic, and this one knows the ORDER the arithmetic happens in.
# ---- tiers: the storage and transfer tier shared by the models (VRAM slots, RAM and file source, peer card) ----
add_library(tiers STATIC src/core/expert_source.cpp src/core/expert_cache.cpp src/core/peer_experts.cpp src/core/conversation_store.cpp src/core/conversation_memory.cpp)
target_include_directories(tiers PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tiers PUBLIC core kernels kernels_cpu ${_gpu_runtime_target})
# the routed-expert tier: slots, per-layer LRU, RAM tier with CPU compute, file reads, second cards
add_library(expert_tier STATIC src/core/expert_tier.cpp)
target_include_directories(expert_tier PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_gpu_include_directories})
target_link_libraries(expert_tier PUBLIC tiers peer_share mmq core kernels kernels_cpu platform ${_gpu_runtime_target})
if(BUILD_TESTING)

  add_executable(host_registration_test tests/core/host_registration_test.cpp)
  target_link_libraries(host_registration_test PRIVATE tiers)
  basalt_embed_manifest(host_registration_test)
  basalt_gpu_test(NAME host_registration_test COMMAND host_registration_test)
  set_tests_properties(host_registration_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # Tests of the shared tiers that upstream built only with qwen4exp; they link the shared libraries here.
  # the file and RAM expert source: reads land every blob byte for byte (no GPU)
  add_executable(file_expert_source_test tests/core/file_expert_source_test.cpp)
  target_link_libraries(file_expert_source_test PRIVATE tiers)
  basalt_embed_manifest(file_expert_source_test)
  basalt_cpu_test(NAME file_expert_source_test COMMAND file_expert_source_test)
  set_tests_properties(file_expert_source_test PROPERTIES TIMEOUT 60)
  # the learned expert profile: its order and file format (CPU only)
  add_executable(expert_profile_save_test tests/core/expert_profile_save_test.cpp)
  target_link_libraries(expert_profile_save_test PRIVATE tiers)
  basalt_embed_manifest(expert_profile_save_test)
  basalt_cpu_test(NAME expert_profile_save_test COMMAND expert_profile_save_test)
  set_tests_properties(expert_profile_save_test PROPERTIES TIMEOUT 60)
  # the expert layout text: real models' layouts and the refusals
  add_executable(expert_layout_test tests/core/expert_layout_test.cpp)
  target_link_libraries(expert_layout_test PRIVATE tiers)
  basalt_cpu_test(NAME expert_layout_test COMMAND expert_layout_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/native_experts)
  set_tests_properties(expert_layout_test PROPERTIES TIMEOUT 60)
  # the .basalt reader: table, meta JSON, the GGUF view of the source tensors, the layout text
  add_executable(model_file_test tests/core/model_file_test.cpp)
  target_link_libraries(model_file_test PRIVATE tiers)
  basalt_embed_manifest(model_file_test)
  basalt_cpu_test(NAME model_file_test COMMAND model_file_test)
  set_tests_properties(model_file_test PROPERTIES TIMEOUT 60)
  # per-layer admission on a sized cache starts each layer in its own range
  add_executable(expert_cache_per_layer_test tests/core/expert_cache_per_layer_test.cpp)
  target_link_libraries(expert_cache_per_layer_test PRIVATE tiers)
  basalt_gpu_test(NAME expert_cache_per_layer_test COMMAND expert_cache_per_layer_test)
  set_tests_properties(expert_cache_per_layer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  add_executable(expert_cache_segmented_test tests/core/expert_cache_segmented_test.cpp)
  target_link_libraries(expert_cache_segmented_test PRIVATE tiers)
  basalt_gpu_test(NAME expert_cache_segmented_test COMMAND expert_cache_segmented_test)
  set_tests_properties(expert_cache_segmented_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
  add_executable(file_expert_homes_test tests/core/file_expert_homes_test.cpp)
  target_link_libraries(file_expert_homes_test PRIVATE tiers)
  basalt_embed_manifest(file_expert_homes_test)
  basalt_gpu_test(NAME file_expert_homes_test COMMAND file_expert_homes_test)
  set_tests_properties(file_expert_homes_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  add_executable(file_expert_repin_test tests/core/file_expert_repin_test.cpp)
  target_link_libraries(file_expert_repin_test PRIVATE tiers)
  basalt_embed_manifest(file_expert_repin_test)
  basalt_gpu_test(NAME file_expert_repin_test COMMAND file_expert_repin_test)
  set_tests_properties(file_expert_repin_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
  # a second card's share of a layer's experts against the main card's grouped kernels, bit for bit (two GPUs)
  add_executable(ds41_peer_parity tests/kernels/ds41_peer_parity.cpp)
  target_include_directories(ds41_peer_parity PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(ds41_peer_parity PRIVATE tiers kernels kernels_cpu core ${_gpu_runtime_target})
  basalt_gpu_test(NAME ds41_peer_parity COMMAND ds41_peer_parity)
  # MMQ for the IQ expert formats against a double-precision reference (--bench: timings)
  add_executable(prefill_mmq_iq1s_test tests/cuda/prefill_mmq_iq1s_test.cpp)
  target_link_libraries(prefill_mmq_iq1s_test PRIVATE mmq kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_mmq_iq1s_test COMMAND prefill_mmq_iq1s_test)
  set_tests_properties(prefill_mmq_iq1s_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # the TMA dense W4A4 product against a double reference, rows bitwise at any chunk length; --bench for TFLOP/s
  add_executable(fp4_dense_tma_test tests/kernels/fp4_dense_tma_test.cpp)
  target_link_libraries(fp4_dense_tma_test PRIVATE prefill_fp4 kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME fp4_dense_tma_test COMMAND fp4_dense_tma_test)
  set_tests_properties(fp4_dense_tma_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # the SwiGLU epilogue's branch-free SiLU quotient against the plain expression on every float
  add_executable(silu_parity tests/kernels/silu_parity.cu)
  set_target_properties(silu_parity PROPERTIES CUDA_ARCHITECTURES 120a-real)
  target_include_directories(silu_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(silu_parity PRIVATE ${_gpu_runtime_target})
  basalt_gpu_test(NAME silu_parity COMMAND silu_parity)
  set_tests_properties(silu_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
if(BUILD_TESTING)
  # the conversation cache: retention bookkeeping (header-only), the memory budget, the snapshot's validation (host
  # only: CUDA is linked but never initialized) and its transfer-failure paths, and a snapshot round trip on a device
  add_executable(conversation_cache_test tests/core/conversation_cache_test.cpp)
  target_include_directories(conversation_cache_test PRIVATE include)
  basalt_cpu_test(NAME conversation_cache_test COMMAND conversation_cache_test)
  # RAM pressure on the parked conversations: the release policy on fake readings (header-only)
  add_executable(conversation_pressure_test tests/core/conversation_pressure_test.cpp)
  target_include_directories(conversation_pressure_test PRIVATE include)
  basalt_cpu_test(NAME conversation_pressure_test COMMAND conversation_pressure_test)
  # conversation_memory.cpp asks the platform layer for the free RAM, so the probe comes along; it needs the system
  # allocator and the OS memory query, and nothing from the GPU runtime.
  add_executable(conversation_memory_test tests/core/conversation_memory_test.cpp src/core/conversation_memory.cpp)
  target_include_directories(conversation_memory_test PRIVATE include)
  target_link_libraries(conversation_memory_test PRIVATE platform)
  basalt_cpu_test(NAME conversation_memory_test COMMAND conversation_memory_test)
  add_executable(checkpoint_transfer_test tests/core/checkpoint_transfer_test.cpp)
  target_link_libraries(checkpoint_transfer_test PRIVATE core)
  basalt_gpu_test(NAME checkpoint_transfer_test COMMAND checkpoint_transfer_test)
  set_tests_properties(checkpoint_transfer_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

# `basalt` (generate.cpp): the engine program that composes the components into a token.  The prompt arrives as
# ids, so a tokenizer is never on one side of a logit comparison.
# The prompt path: batched prompt processing (tensor-core GEMMs through cuBLAS).

add_library(serve_shared STATIC src/program/serve/commands.cpp
            src/program/probe_rates.cpp
            src/program/serve/protocol.cpp
            src/program/serve/runtime.cpp
            src/program/serve/request.cpp
            src/program/serve/request_finish.cpp
            src/program/serve/request_image.cpp
            src/program/serve/request_parse.cpp
            src/program/serve/request_prefill.cpp
            src/program/serve/request_restore.cpp
            src/program/serve/prompt_handles.cpp
            src/program/serve/session_engine.cpp
            src/program/cli/tokens.cpp
            src/program/cli/help.cpp)
target_include_directories(serve_shared PUBLIC include PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src/program/tokenizer)
target_link_libraries(serve_shared PUBLIC tiers model engine_session grammar tokenizer spec ${_gpu_runtime_target})

add_executable(basalt src/program/generate.cpp
                       src/program/generate/adapter.cpp
                       src/program/generate/configure.cpp
                       src/program/cli/arguments.cpp
                       src/program/cli/serve.cpp
                       src/program/cli/startup.cpp
                       src/program/cli/gpu.cpp)
basalt_ship_to_bin(basalt)
target_include_directories(basalt PRIVATE ${_gpu_include_directories})
target_link_libraries(basalt PRIVATE models serve_shared engine_session ${_gpu_runtime_target})
basalt_embed_manifest(basalt)
# `basalt serve` runs the Python server from the directory above bin/; the engine reads its expert profiles from data/profiles

# The standard benchmark: a host-only program that drives the engine over its stdin/stdout protocol.
# The commit stamp is refreshed on every build (not only at configure time), by a target that rewrites the header
# only when the commit changed.
set(_git_stamp ${CMAKE_BINARY_DIR}/stamp/basalt_git.h)
add_custom_target(git_stamp
  COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${_git_stamp} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_stamp.cmake
  BYPRODUCTS ${_git_stamp}
  COMMENT "Refreshing the commit stamp")
execute_process(COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${_git_stamp}
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/git_stamp.cmake)
# the engine driven as a child process and the report logic, shared by basalt-bench and basalt-probe
add_library(engine_session STATIC src/program/engine_session.cpp src/program/load_summary.cpp src/program/probe_report.cpp src/program/probe_derive.cpp src/program/auto_plan.cpp src/program/community_row.cpp src/program/host_rates.cpp
            src/program/gpu_select.cpp src/program/multi_gpu_plan.cpp src/program/image_loan.cpp)
target_include_directories(engine_session PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(engine_session PUBLIC platform kernels_cpu Threads::Threads)   # the pool's sizing rule (auto_pool_workers)
# the GBNF grammar engine and the JSON-schema and tool-call grammar builders (CPU only, no engine dependency)
add_library(grammar STATIC src/program/grammar/grammar.cpp src/program/grammar/schema.cpp src/program/grammar/json_schema.cpp
            src/program/grammar/schema_convert.cpp src/program/grammar/guard.cpp)
target_include_directories(grammar PUBLIC include)
target_link_libraries(grammar PRIVATE warnings)

# the C++ byte-level BPE tokenizer the engine runs on request text (CPU only)
add_library(tokenizer STATIC src/program/tokenizer/bpe.cpp src/program/tokenizer/unicode_flags.cpp)
target_include_directories(tokenizer PUBLIC include PRIVATE src/program/tokenizer)
target_link_libraries(tokenizer PRIVATE warnings)
if(BUILD_TESTING)
  # built from the tokenizer sources with the standard library range checks on, so an out-of-range index aborts the test;
  # the synthetic cases split with a reference pre-tokenizer kept beside the test, a pack's parity with Hy4's own rule
  add_executable(tokenizer_test tests/program/tokenizer_test.cpp tests/program/pretok_qwen35.cpp src/program/tokenizer/bpe.cpp
                 src/program/tokenizer/unicode_flags.cpp src/models/hy4flash/pretok_hy4.cpp)
  target_include_directories(tokenizer_test PRIVATE include src src/program/tokenizer)
  target_compile_definitions(tokenizer_test PRIVATE _GLIBCXX_ASSERTIONS)
  target_link_libraries(tokenizer_test PRIVATE warnings)
  basalt_embed_manifest(tokenizer_test)
  add_executable(tokenizer_bench tests/program/tokenizer_bench.cpp src/models/hy4flash/pretok_hy4.cpp)
  target_include_directories(tokenizer_bench PRIVATE src src/program/tokenizer)
  target_link_libraries(tokenizer_bench PRIVATE tokenizer warnings)
  basalt_embed_manifest(tokenizer_bench)
  basalt_cpu_test(NAME tokenizer_test COMMAND tokenizer_test)
  set_tests_properties(tokenizer_test PROPERTIES TIMEOUT 60)
endif()
add_executable(basalt-bench src/program/bench_main.cpp)
basalt_ship_to_bin(basalt-bench)
# the image encoder sidecar, bin/basalt-vision (its own project, built with this one)
add_subdirectory(tools/vision)
target_link_libraries(serve_shared PUBLIC vision_core)
if(BUILD_TESTING)
  # the host half of the vision library needs no GPU: the decode guard is checked here
  add_executable(vision_preprocess_test tests/vision/preprocess_limit_test.cpp)
  target_link_libraries(vision_preprocess_test PRIVATE vision_core)
  basalt_cpu_test(NAME vision_preprocess_test COMMAND vision_preprocess_test)
  # the image grid and canvas against the reference processor's sizes and Pillow's pixels (host only)
  add_executable(vision_resize_test tests/vision/preprocess_test.cpp)
  target_link_libraries(vision_resize_test PRIVATE vision_core warnings)
  basalt_cpu_test(NAME vision_resize_test COMMAND vision_resize_test)
  basalt_gpu_test(NAME cls_encoder_parity COMMAND cls_encoder_parity)
  # FFmpeg found at run time, or the reason it is not (any machine: CI has no FFmpeg)
  add_executable(ffmpeg_load_test tests/vision/ffmpeg_load_test.cpp)
  target_link_libraries(ffmpeg_load_test PRIVATE vision_core warnings)
  basalt_embed_manifest(ffmpeg_load_test)
  basalt_cpu_test(NAME ffmpeg_load_test COMMAND ffmpeg_load_test)
  # video sampling, canvas, timestamps and patches against the reference processor; the committed clip is decoded
  # where this machine has FFmpeg
  add_executable(video_test tests/vision/video_test.cpp)
  target_link_libraries(video_test PRIVATE vision_core warnings)
  basalt_embed_manifest(video_test)
  basalt_cpu_test(NAME video_test COMMAND video_test ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/video/synth.mp4)
  # a prompt's rotary positions with pictures and video groups against the reference's get_rope_index
  add_executable(mrope_positions_test tests/vision/mrope_positions_test.cpp)
  target_link_libraries(mrope_positions_test PRIVATE vision_core warnings)
  basalt_cpu_test(NAME mrope_positions_test COMMAND mrope_positions_test)
  # the hand-declared FFmpeg ABI against FFmpeg's own headers, where this machine has them (nothing is linked)
  find_path(BASALT_FFMPEG_HEADERS libavformat/avformat.h)
  if(BASALT_FFMPEG_HEADERS)
    add_executable(ffmpeg_abi_test tests/vision/ffmpeg_abi_test.cpp)
    target_include_directories(ffmpeg_abi_test PRIVATE include ${BASALT_FFMPEG_HEADERS})
    basalt_cpu_test(NAME ffmpeg_abi_test COMMAND ffmpeg_abi_test)
    set_tests_properties(ffmpeg_abi_test PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()
add_dependencies(basalt-bench basalt basalt-device git_stamp)
add_dependencies(serve_shared git_stamp)   # the engine's INFO commit=
target_include_directories(serve_shared PRIVATE ${CMAKE_BINARY_DIR}/stamp)
target_include_directories(basalt-bench PRIVATE ${CMAKE_BINARY_DIR}/stamp)
target_link_libraries(basalt-bench PRIVATE engine_session platform warnings ${CMAKE_DL_LIBS})
basalt_embed_manifest(basalt-bench)
target_compile_definitions(basalt-bench PRIVATE
  BASALT_BENCH_ENGINE="$<TARGET_FILE:basalt>"
  BASALT_BENCH_DEVICE="$<TARGET_FILE:basalt-device>"
  BASALT_BENCH_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/data/bench"
  BASALT_BENCH_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  BASALT_BENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  BASALT_BENCH_CUDA="${CMAKE_CUDA_COMPILER_VERSION}")

# basalt-probe: starts the engine, measures its rates and cold decode speed, prints a report
add_executable(basalt-probe src/program/probe_main.cpp)
basalt_ship_to_bin(basalt-probe)
add_dependencies(basalt-probe basalt)
target_link_libraries(basalt-probe PRIVATE engine_session platform warnings ${CMAKE_DL_LIBS})
basalt_embed_manifest(basalt-probe)
target_compile_definitions(basalt-probe PRIVATE
  BASALT_BENCH_ENGINE="$<TARGET_FILE:basalt>"
  BASALT_BENCH_FIXTURES="${CMAKE_CURRENT_SOURCE_DIR}/data/bench")

if(BUILD_TESTING)
# closing the engine's stdin mid-request stops it (runs the real engine on a model pack:
# -DBASALT_STDIN_STOP_MODEL=<pack>, with python3)
if(BASALT_STDIN_STOP_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME stdin_stop_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/stdin_stop_test.py
                   $<TARGET_FILE:basalt> ${BASALT_STDIN_STOP_MODEL} ${CMAKE_CURRENT_SOURCE_DIR}/data/bench/conv.ids
                   ${BASALT_STDIN_STOP_ARGS})
  set_property(TEST stdin_stop_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(stdin_stop_test PROPERTIES TIMEOUT 600)
endif()

# the engine enforces a request grammar on what it samples (runs the real engine on a model pack:
# -DBASALT_GRAMMAR_MODEL=<pack>, with python3)
if(BASALT_GRAMMAR_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME grammar_engine_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/grammar_engine_test.py
                   $<TARGET_FILE:basalt> ${BASALT_GRAMMAR_MODEL} ${BASALT_GRAMMAR_ARGS})
  set_property(TEST grammar_engine_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(grammar_engine_test PROPERTIES TIMEOUT 900)
endif()

# grammar requests under --batch: each slot held to its own grammar (starts the server on a model pack, same
# -DBASALT_GRAMMAR_MODEL, with python3; -DBASALT_GRAMMAR_BATCH_ARGS=<server args>)
if(BASALT_GRAMMAR_MODEL AND Python3_Interpreter_FOUND)
  add_test(NAME grammar_batch_test
           COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/grammar_batch_test.py
                   $<TARGET_FILE:basalt> ${BASALT_GRAMMAR_MODEL} ${BASALT_GRAMMAR_BATCH_ARGS})
  set_property(TEST grammar_batch_test APPEND PROPERTY LABELS gpu)
  set_tests_properties(grammar_batch_test PROPERTIES TIMEOUT 1500)
endif()

# the route tail skip against a plain CPU rule: which entries it drops over random windows, tables and ranks at each MoE
# model's geometry (GPU, synthetic, no model; --bench times the window kernel)
add_executable(route_tail_skip_parity tests/kernels/route_tail_skip_parity.cpp)
target_include_directories(route_tail_skip_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(route_tail_skip_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME route_tail_skip_parity COMMAND route_tail_skip_parity)
set_tests_properties(route_tail_skip_parity PROPERTIES SKIP_RETURN_CODE 77)
add_executable(dense_window_bench tests/kernels/dense_window_bench.cpp)
target_link_libraries(dense_window_bench PRIVATE kernels mmq ${_gpu_runtime_target})
add_executable(dense_ops_parity tests/kernels/dense_ops_parity.cpp)
target_link_libraries(dense_ops_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME dense_ops_parity COMMAND dense_ops_parity)
set_tests_properties(dense_ops_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_executable(nvfp4_planes_test tests/kernels/nvfp4_planes_test.cpp)
target_link_libraries(nvfp4_planes_test PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME nvfp4_planes_test COMMAND nvfp4_planes_test)
set_tests_properties(nvfp4_planes_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_executable(gdn_seg_split_parity tests/kernels/gdn_seg_split_parity.cpp)
target_link_libraries(gdn_seg_split_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME gdn_seg_split_parity COMMAND gdn_seg_split_parity)
set_tests_properties(gdn_seg_split_parity PROPERTIES SKIP_RETURN_CODE 77)
add_executable(window_gemm_parity tests/kernels/window_gemm_parity.cpp)
target_link_libraries(window_gemm_parity PRIVATE window_gemm kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME window_gemm_parity COMMAND window_gemm_parity)
set_tests_properties(window_gemm_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
# a second card planned as the main card gives its window products bit for bit (two GPUs; skipped without)
add_executable(window_gemm_two_gpu tests/kernels/window_gemm_two_gpu.cpp)
target_link_libraries(window_gemm_two_gpu PRIVATE window_gemm kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME window_gemm_two_gpu COMMAND window_gemm_two_gpu)
set_tests_properties(window_gemm_two_gpu PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
# the FP8 and FP4 codes as bf16x2 (narrow_bf16.cuh): the direct cvt (CUDA 13.2+) and the f16 route, every input
add_executable(narrow_bf16_parity tests/kernels/narrow_bf16_parity.cu)
set_target_properties(narrow_bf16_parity PROPERTIES CUDA_ARCHITECTURES 120a-real)
target_include_directories(narrow_bf16_parity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(narrow_bf16_parity PRIVATE ${_gpu_runtime_target})
basalt_gpu_test(NAME narrow_bf16_parity COMMAND narrow_bf16_parity)
set_tests_properties(narrow_bf16_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_executable(gqa_attn_parity tests/kernels/gqa_attn_parity.cpp)
target_link_libraries(gqa_attn_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME gqa_attn_parity COMMAND gqa_attn_parity)
set_tests_properties(gqa_attn_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
# a cache split in two places (K/V streaming): every attention kernel and the cache writes bitwise their one-buffer launches
add_executable(gqa_far_parity tests/kernels/gqa_far_parity.cpp)
target_link_libraries(gqa_far_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME gqa_far_parity COMMAND gqa_far_parity)
set_tests_properties(gqa_far_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_executable(iq_parity tests/kernels/iq_parity.cpp)
target_include_directories(iq_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(iq_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_embed_manifest(iq_parity)
# the i-quant fixtures are generated, not shipped.  tools/iq_fixture.py needs python3 + numpy
# and the repository's vendored gguf-py (no pip installs); it writes deterministic fixtures into the
# BUILD directory (the source tree is not touched).  The CTest fixture makes generation-to-test order
# explicit even under parallel execution; a fixture-generation failure fails iq_parity with it.
find_package(Python3 COMPONENTS Interpreter QUIET)

# appending a community row to docs/community/benchmarks.json: formatting cases, and a byte comparison with
# tools/check_benchmarks.py --fix (python3)
add_executable(community_row_test tests/program/community_row_test.cpp)
target_link_libraries(community_row_test PRIVATE engine_session platform warnings)
basalt_embed_manifest(community_row_test)
basalt_cpu_test(NAME community_row_test COMMAND community_row_test)
set_tests_properties(community_row_test PROPERTIES TIMEOUT 60)
if(Python3_Interpreter_FOUND)
  if(BASALT_MODEL_HY4FLASH)
  # a long prompt's last-row logits in one chunk and in chunks of one attention step each
  basalt_gpu_test(NAME hy4flash_prompt_steps_test
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/models/hy4flash/prompt_steps_test.py
                          --engine $<TARGET_FILE:basalt>
                          --model ${CMAKE_CURRENT_SOURCE_DIR}/models/Hy4-flash-NVFP4.basalt)
  set_tests_properties(hy4flash_prompt_steps_test PROPERTIES TIMEOUT 1800 SKIP_RETURN_CODE 77)
  endif()
  basalt_cpu_test(NAME community_row_fix_test
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/community_row_fix_test.py
                          $<TARGET_FILE:community_row_test> ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(community_row_fix_test PROPERTIES TIMEOUT 60)
  # protocol lines written from several threads each arrive whole on stdout
  add_executable(protocol_test tests/core/protocol_test.cpp)
  target_include_directories(protocol_test PRIVATE include)
  target_link_libraries(protocol_test PRIVATE warnings Threads::Threads)
  basalt_embed_manifest(protocol_test)
  basalt_cpu_test(NAME protocol_test
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/core/protocol_test.py
                          $<TARGET_FILE:protocol_test>)
  set_tests_properties(protocol_test PROPERTIES TIMEOUT 60)
endif()

# the C++ tokenizer's ids against tools/tokenizer.py over a corpus
# (-DBASALT_TOKENIZER_DIR=<a pack's extracted tokenizer/ directory>, with python3 and the regex package)
if(BASALT_TOKENIZER_DIR AND Python3_Interpreter_FOUND)
  basalt_cpu_test(NAME tokenizer_parity
                  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/program/tokenizer_parity.py
                          $<TARGET_FILE:tokenizer_test> ${BASALT_TOKENIZER_DIR} ${BASALT_TOKENIZER_BIG})
  set_tests_properties(tokenizer_parity PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 77)
endif()
# the generator needs an interpreter WITH numpy: probe the configured one, a .venv, and the
# system python3; -DIQ_FIXTURE_PYTHON=... overrides the search outright.
set(_iq_python "")
set(_iq_candidates "")
if(IQ_FIXTURE_PYTHON)
  list(APPEND _iq_candidates ${IQ_FIXTURE_PYTHON})
else()
  if(Python3_Interpreter_FOUND)
    list(APPEND _iq_candidates ${Python3_EXECUTABLE})
  endif()
  list(APPEND _iq_candidates "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python"
                              /usr/bin/python3
                              python3)
endif()
foreach(_cand IN LISTS _iq_candidates)
  if(_cand STREQUAL "")
    continue()
  endif()
  execute_process(COMMAND ${_cand} -c "import numpy" OUTPUT_QUIET ERROR_QUIET RESULT_VARIABLE _iq_probe)
  if(_iq_probe EQUAL 0)
    set(_iq_python "${_cand}")
    break()
  endif()
endforeach()
if(NOT _iq_python STREQUAL "")
  basalt_cpu_test(NAME iq_parity_fixtures
           COMMAND ${_iq_python} ${CMAKE_CURRENT_SOURCE_DIR}/tools/iq_fixture.py
                   --out ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
  # exit 3: numpy or the vendored gguf-py is missing (a trimmed checkout) - a skip, not a failure
  set_tests_properties(iq_parity_fixtures PROPERTIES FIXTURES_SETUP iq_fixture_data SKIP_RETURN_CODE 3)
  basalt_gpu_test(NAME iq_parity COMMAND iq_parity ${CMAKE_CURRENT_BINARY_DIR}/iq_fixture)
  set_tests_properties(iq_parity PROPERTIES FIXTURES_REQUIRED iq_fixture_data SKIP_RETURN_CODE 3)
  # the converter's own tests (the checkpoint repack, the pack sections, the requantizers): plain unittest, host only
  basalt_cpu_test(NAME convert_tests COMMAND ${_iq_python} -m unittest discover -s tools/convert -p "test_*.py"
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(convert_tests PROPERTIES TIMEOUT 300)
  # the v1 -> v2 file migration on a tiny file
  basalt_cpu_test(NAME migration_tests COMMAND ${_iq_python} -m unittest discover -s tools/migrations -p "test_*.py"
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
else()
  message(STATUS "basalt tests: no python3 with numpy was found (the configured interpreter and .venv "
                 "were probed) - the iq_parity fixtures and its test and the converter tests are not registered; "
                 "they need python3 with numpy and gguf-py")
endif()
# native_mmvq's multi_exact contract, bitwise: a T-column call against T single-column calls on the same Q8_1
# bytes, with the flag off as the negative control (synthetic, GPU, no model; the control assumes 32-lane warps)
add_executable(mmvq_multi_parity tests/kernels/mmvq_multi_parity.cpp)
target_include_directories(mmvq_multi_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(mmvq_multi_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME mmvq_multi_parity COMMAND mmvq_multi_parity)
# the draft head screen (q80_screen) against native_mmvq's one-column logits and their argmax, and against a CPU reference
# (synthetic at the real width, GPU, no model; --bench adds the microbench)
add_executable(q80_screen_parity tests/kernels/q80_screen_parity.cpp)
target_include_directories(q80_screen_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(q80_screen_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME q80_screen_parity COMMAND q80_screen_parity)
# the IQ kernels that decode a weight part once for every column/entry, bitwise vs the per-column ones
# (GPU, synthetic, no model; --bench adds old/new timings)
add_executable(iq_multi_parity tests/kernels/iq_multi_parity.cpp)
target_include_directories(iq_multi_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(iq_multi_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME iq_multi_parity COMMAND iq_multi_parity)

add_executable(sampler_parity tests/kernels/sampler_parity.cpp)
target_include_directories(sampler_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(sampler_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME sampler_parity COMMAND sampler_parity --selftest)
basalt_gpu_test(NAME sampler_split_parity COMMAND sampler_parity --split)
basalt_gpu_test(NAME sampler_drawn_parity COMMAND sampler_parity --drawn)
# a pick's log probability and the row's likeliest tokens (row_logprobs) against a double CPU reference
add_executable(logprobs_parity tests/kernels/logprobs_parity.cpp)
target_include_directories(logprobs_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(logprobs_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME logprobs_parity COMMAND logprobs_parity)

add_executable(quantize_act_parity tests/kernels/quantize_act_parity.cpp)
target_include_directories(quantize_act_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(quantize_act_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME quantize_act_parity COMMAND quantize_act_parity --selftest)

add_executable(elementwise_parity tests/kernels/elementwise_parity.cpp)
target_include_directories(elementwise_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(elementwise_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME elementwise_parity COMMAND elementwise_parity --selftest)

add_executable(bf16_gemv_parity tests/kernels/bf16_gemv_parity.cpp)
target_include_directories(bf16_gemv_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(bf16_gemv_parity PRIVATE kernels ${_gpu_runtime_target})
basalt_gpu_test(NAME bf16_gemv_parity COMMAND bf16_gemv_parity --selftest)

add_executable(dequant_f16_parity tests/kernels/dequant_f16_parity.cpp)
target_include_directories(dequant_f16_parity PRIVATE ${_gpu_include_directories})
target_link_libraries(dequant_f16_parity PRIVATE kernels artifact ${_gpu_runtime_target})

# working-set lock of a host region (CPU only).
add_executable(platform_memory_test tests/platform/memory_test.cpp)
target_link_libraries(platform_memory_test PRIVATE core ${_gpu_runtime_target})
basalt_embed_manifest(platform_memory_test)
basalt_cpu_test(NAME platform_memory_test COMMAND platform_memory_test)

# A path carried as UTF-8 reaching the OS intact: the seam in both directions, and a path past 260 chars
# (longPathAware). The program also runs itself as a child with a non-ASCII name as one argument, which is how a
# pack path typed by a user arrives; on Windows argv is decoded with the process code page, so that case only
# passes because this executable carries the application manifest.
add_executable(path_utf8_test tests/platform/path_utf8_test.cpp)
target_link_libraries(path_utf8_test PRIVATE platform)
basalt_embed_manifest(path_utf8_test)
basalt_cpu_test(NAME path_utf8_test COMMAND path_utf8_test)
set_tests_properties(path_utf8_test PROPERTIES TIMEOUT 60)

# platform::MappedFile, file_size and pread_all on a scratch file (CPU only).
add_executable(mapped_file_test tests/platform/mapped_file_test.cpp)
target_link_libraries(mapped_file_test PRIVATE platform)
basalt_embed_manifest(mapped_file_test)
basalt_cpu_test(NAME mapped_file_test COMMAND mapped_file_test)
set_tests_properties(mapped_file_test PROPERTIES TIMEOUT 60)
if(NOT WIN32)
  add_executable(mapped_file_large_test tests/platform/mapped_file_large_test.cpp)
  target_link_libraries(mapped_file_large_test PRIVATE platform)
  basalt_cpu_test(NAME mapped_file_large_test COMMAND mapped_file_large_test)
  set_tests_properties(mapped_file_large_test PROPERTIES TIMEOUT 60)
endif()

# the platform seams below the engine: thread placement and topology, the compiler helpers, large host regions, the
# piped child, the system questions, the dynamic loader and the file writes and directory calls (CPU only, both platforms).
foreach(_t cpu_test compiler_test region_test process_pipe_test system_test file_write_test)
  add_executable(platform_${_t} tests/platform/${_t}.cpp)
  target_link_libraries(platform_${_t} PRIVATE platform Threads::Threads)
  target_include_directories(platform_${_t} PRIVATE include)
  basalt_embed_manifest(platform_${_t})
  basalt_cpu_test(NAME platform_${_t} COMMAND platform_${_t})
  set_tests_properties(platform_${_t} PROPERTIES TIMEOUT 120)
endforeach()

# the engine's load lines condensed for basalt-bench: lines, throttling, terminal mode (CPU only)
add_executable(load_summary_test tests/program/load_summary_test.cpp)
target_link_libraries(load_summary_test PRIVATE engine_session warnings)
basalt_embed_manifest(load_summary_test)
basalt_cpu_test(NAME load_summary_test COMMAND load_summary_test)
set_tests_properties(load_summary_test PROPERTIES TIMEOUT 60)

# the engine as a child process: an error line in a healthy start does not fail it, an exit before READY does (CPU only)
add_executable(engine_session_test tests/program/engine_session_test.cpp)
target_link_libraries(engine_session_test PRIVATE engine_session warnings)
basalt_embed_manifest(engine_session_test)
basalt_cpu_test(NAME engine_session_test COMMAND engine_session_test)
set_tests_properties(engine_session_test PROPERTIES TIMEOUT 60)

# the engine log helper: the line each level renders (CPU only)
add_executable(log_test tests/core/log_test.cpp)
target_include_directories(log_test PRIVATE include)
target_link_libraries(log_test PRIVATE warnings)
basalt_embed_manifest(log_test)
basalt_cpu_test(NAME log_test COMMAND log_test)
set_tests_properties(log_test PROPERTIES TIMEOUT 60)
# the expert tiers' admission and placement rules on synthetic id streams (no GPU)
add_executable(tier_policy_test tests/core/tier_policy_test.cpp)
target_include_directories(tier_policy_test PRIVATE include)
basalt_embed_manifest(tier_policy_test)
basalt_cpu_test(NAME tier_policy_test COMMAND tier_policy_test)
add_executable(file_reads_test tests/core/file_reads_test.cpp src/core/file_reads.cpp)
target_include_directories(file_reads_test PRIVATE include)
target_compile_definitions(file_reads_test PRIVATE BASALT_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(file_reads_test PRIVATE platform Threads::Threads)
basalt_embed_manifest(file_reads_test)
basalt_cpu_test(NAME file_reads_test COMMAND file_reads_test)
# the shared expert tier on a synthetic pack: slots follow a routing stream with an admission threshold (needs a GPU)
add_executable(expert_tier_test tests/core/expert_tier_test.cpp)
target_include_directories(expert_tier_test PRIVATE include ${_gpu_include_directories})
target_compile_definitions(expert_tier_test PRIVATE BASALT_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
target_link_libraries(expert_tier_test PRIVATE expert_tier ${_gpu_runtime_target})
basalt_embed_manifest(expert_tier_test)
basalt_gpu_test(NAME expert_tier_test COMMAND expert_tier_test)
set_tests_properties(expert_tier_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
add_executable(exact_slots_test tests/core/exact_slots_test.cpp)
target_include_directories(exact_slots_test PRIVATE include)
target_link_libraries(exact_slots_test PRIVATE warnings)
basalt_embed_manifest(exact_slots_test)
basalt_cpu_test(NAME exact_slots_test COMMAND exact_slots_test)
set_tests_properties(exact_slots_test PROPERTIES TIMEOUT 60)

# basalt-probe's report: rates block and layout (CPU only)
add_executable(probe_report_test tests/program/probe_report_test.cpp)
target_link_libraries(probe_report_test PRIVATE engine_session warnings)
basalt_embed_manifest(probe_report_test)
basalt_cpu_test(NAME probe_report_test COMMAND probe_report_test)
set_tests_properties(probe_report_test PROPERTIES TIMEOUT 60)

# the gpu setting: parsing, the card choice and the roles, over fake card tables (CPU only)
add_executable(gpu_select_test tests/program/gpu_select_test.cpp)
target_link_libraries(gpu_select_test PRIVATE engine_session warnings)
basalt_embed_manifest(gpu_select_test)
basalt_cpu_test(NAME gpu_select_test COMMAND gpu_select_test)
set_tests_properties(gpu_select_test PROPERTIES TIMEOUT 60)
add_executable(ppl_gate_test tests/program/ppl_gate_test.cpp)
target_link_libraries(ppl_gate_test PRIVATE engine_session warnings)
basalt_embed_manifest(ppl_gate_test)
basalt_cpu_test(NAME ppl_gate_test COMMAND ppl_gate_test)
set_tests_properties(ppl_gate_test PROPERTIES TIMEOUT 60)
add_executable(multi_gpu_plan_test tests/program/multi_gpu_plan_test.cpp)
target_link_libraries(multi_gpu_plan_test PRIVATE engine_session warnings)
basalt_embed_manifest(multi_gpu_plan_test)
basalt_cpu_test(NAME multi_gpu_plan_test COMMAND multi_gpu_plan_test)
set_tests_properties(multi_gpu_plan_test PROPERTIES TIMEOUT 60)

# basalt-probe's suggested values: the worker pick, the balance share and the suggestion lines (CPU only)
add_executable(probe_derive_test tests/program/probe_derive_test.cpp)
target_link_libraries(probe_derive_test PRIVATE engine_session warnings)
basalt_embed_manifest(probe_derive_test)
basalt_cpu_test(NAME probe_derive_test COMMAND probe_derive_test)
set_tests_properties(probe_derive_test PROPERTIES TIMEOUT 60)

# the start-up picks for settings left out: the rules and the printed line (CPU only)
add_executable(image_loan_test tests/program/image_loan_test.cpp)
target_link_libraries(image_loan_test PRIVATE engine_session warnings)
basalt_embed_manifest(image_loan_test)
basalt_cpu_test(NAME image_loan_test COMMAND image_loan_test)
set_tests_properties(image_loan_test PROPERTIES TIMEOUT 60)
# the rows a verify window keeps when it ends the reply or not (CPU only)
add_executable(reply_rows_test tests/program/reply_rows_test.cpp)
target_include_directories(reply_rows_test PRIVATE src/program/serve)
target_link_libraries(reply_rows_test PRIVATE warnings)
basalt_embed_manifest(reply_rows_test)
basalt_cpu_test(NAME reply_rows_test COMMAND reply_rows_test)
set_tests_properties(reply_rows_test PROPERTIES TIMEOUT 60)
# the console's per-slot counts and the expert map's sample (CPU only)
add_executable(console_stats_test tests/program/console_stats_test.cpp)
target_include_directories(console_stats_test PRIVATE src/program/serve include)
target_link_libraries(console_stats_test PRIVATE warnings)
basalt_embed_manifest(console_stats_test)
basalt_cpu_test(NAME console_stats_test COMMAND console_stats_test)
set_tests_properties(console_stats_test PROPERTIES TIMEOUT 60)
# a sent pick's log probability and the row's likeliest tokens (CPU only)
add_executable(logprobs_test tests/program/logprobs_test.cpp)
target_include_directories(logprobs_test PRIVATE src/program/serve include)
target_link_libraries(logprobs_test PRIVATE warnings)
basalt_embed_manifest(logprobs_test)
basalt_cpu_test(NAME logprobs_test COMMAND logprobs_test)
set_tests_properties(logprobs_test PROPERTIES TIMEOUT 60)
# the lines the server reads from the engine: the conversation cache's figures and the context-overflow ERR
add_executable(protocol_lines_test tests/program/protocol_lines_test.cpp)
target_include_directories(protocol_lines_test PRIVATE include)
target_link_libraries(protocol_lines_test PRIVATE warnings)
basalt_embed_manifest(protocol_lines_test)
basalt_cpu_test(NAME protocol_lines_test COMMAND protocol_lines_test)
set_tests_properties(protocol_lines_test PROPERTIES TIMEOUT 60)
add_executable(auto_plan_test tests/program/auto_plan_test.cpp)
target_link_libraries(auto_plan_test PRIVATE engine_session warnings)
basalt_embed_manifest(auto_plan_test)
basalt_cpu_test(NAME auto_plan_test COMMAND auto_plan_test)
set_tests_properties(auto_plan_test PROPERTIES TIMEOUT 60)

# the grammar engine: parse, accept/reject, UTF-8 across tokens, clone, lazy triggers and the vocabulary mask (CPU only)
add_executable(grammar_test tests/program/grammar_test.cpp)
target_link_libraries(grammar_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_test)
basalt_cpu_test(NAME grammar_test COMMAND grammar_test)
set_tests_properties(grammar_test PROPERTIES TIMEOUT 120)

# the pack vocabulary reader and the per-request window check (CPU only)
add_executable(grammar_guard_test tests/program/grammar_guard_test.cpp)
target_link_libraries(grammar_guard_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_guard_test)
basalt_cpu_test(NAME grammar_guard_test COMMAND grammar_guard_test)
set_tests_properties(grammar_guard_test PROPERTIES TIMEOUT 120)

# schema-to-grammar and the tool-call grammar (CPU only)
add_executable(grammar_schema_test tests/program/grammar_schema_test.cpp)
target_link_libraries(grammar_schema_test PRIVATE grammar warnings)
basalt_embed_manifest(grammar_schema_test)
basalt_cpu_test(NAME grammar_schema_test COMMAND grammar_schema_test)
set_tests_properties(grammar_schema_test PROPERTIES TIMEOUT 120)

# llama.cpp's own grammar tests (parser, accept/reject, state, schema-to-grammar) run against the grammar library (CPU only)
foreach(_t grammar_llama_parser_test grammar_llama_state_test grammar_llama_accept_test grammar_llama_schema_test)
  add_executable(${_t} tests/program/${_t}.cpp)
  target_link_libraries(${_t} PRIVATE grammar warnings)
  basalt_embed_manifest(${_t})
  basalt_cpu_test(NAME ${_t} COMMAND ${_t})
  set_tests_properties(${_t} PROPERTIES TIMEOUT 300)
endforeach()
target_include_directories(grammar_llama_schema_test PRIVATE src/program/grammar)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_executable(pinned_shared_test tests/core/pinned_shared_test.cpp)
  target_link_libraries(pinned_shared_test PRIVATE core ${_gpu_runtime_target})
  basalt_cpu_test(NAME pinned_shared_test COMMAND pinned_shared_test)
endif()

add_executable(loader_test tests/core/loader_test.cpp)
target_link_libraries(loader_test PRIVATE core ${_gpu_runtime_target})
basalt_cpu_test(NAME loader_test COMMAND loader_test)
add_executable(load_progress_test tests/core/load_progress_test.cpp)
target_link_libraries(load_progress_test PRIVATE platform warnings)
target_include_directories(load_progress_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
basalt_cpu_test(NAME load_progress_test COMMAND load_progress_test)
add_executable(device_memory_test tests/core/device_memory_test.cpp)
target_link_libraries(device_memory_test PRIVATE core ${_gpu_runtime_target})
basalt_gpu_test(NAME device_memory_test COMMAND device_memory_test)
set_tests_properties(device_memory_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
add_executable(shipped_profiles_test tests/core/shipped_profiles_test.cpp)
target_link_libraries(shipped_profiles_test PRIVATE platform)
target_include_directories(shipped_profiles_test PRIVATE include)
basalt_cpu_test(NAME shipped_profiles_test COMMAND shipped_profiles_test)

# The GPU test runs ON THE TARGET PC. `--selftest` allocates
# and writes real VRAM, so a machine without a device fails this test rather than skipping it silently -
# which is correct: an engine that targets one GPU should not report a pass for having no GPU.
basalt_gpu_test(NAME cuda_device_selftest COMMAND basalt-device --selftest)
endif()

# ---- kernels_cpu ----
#
# A SEPARATE LIBRARY from `kernels`, and the AVX-512 flags are set on the SOURCE rather than on the target
# so that they apply only to the TU that needs them.  Both parts matter:
#
#   * the CPU expert kernel uses `_mm256_dpbusd_epi32` (AVX512-VNNI) and `_mm256_multishift_epi64_epi8`
#     (AVX512-VBMI);
#   * a TU compiled WITH AVX-512 may use AVX-512 in ANY of its code, so the scalar fallback and the
#     test drivers must NOT be in it - otherwise the fallback is not a fallback, it is the same bet with
#     worse opt-out.  `cpu_require_expert_support()` therefore refuses at startup on a CPU without the
#     features, and the fallback exists for tests.
# speculation components that need no GPU.
add_library(spec STATIC src/spec/suffix_drafter.cpp src/spec/controller.cpp src/spec/draft_policy.cpp)
target_include_directories(spec PUBLIC include)
if(BUILD_TESTING)
  add_executable(suffix_drafter_test tests/spec/suffix_drafter_test.cpp)
  target_link_libraries(suffix_drafter_test PRIVATE spec)
  basalt_cpu_test(NAME suffix_drafter_test COMMAND suffix_drafter_test)
  add_executable(controller_test tests/spec/controller_test.cpp)
  target_link_libraries(controller_test PRIVATE spec)
  basalt_cpu_test(NAME controller_test COMMAND controller_test)
  add_executable(draft_policy_test tests/spec/draft_policy_test.cpp)
  target_link_libraries(draft_policy_test PRIVATE spec)
  basalt_cpu_test(NAME draft_policy_test COMMAND draft_policy_test)
endif()
if(BUILD_TESTING)
  # the serve loop's conversation-cache retention policy (header-only, no library to link)
  add_executable(conv_cache_test tests/program/conv_cache_test.cpp)
  target_include_directories(conv_cache_test PRIVATE include)
  basalt_cpu_test(NAME conv_cache_test COMMAND conv_cache_test)
  # a held conversation met against a prompt, the model's own pieces standing for the same bytes (header-only, CPU)
  add_executable(prefix_anchor_test tests/program/prefix_anchor_test.cpp)
  target_include_directories(prefix_anchor_test PRIVATE src)
  target_link_libraries(prefix_anchor_test PRIVATE warnings)
  basalt_cpu_test(NAME prefix_anchor_test COMMAND prefix_anchor_test)
  # where a prompt read from token 0 places the chain's root: the system prompt's end, never the reply's header (CPU)
  add_executable(prompt_parts_test tests/program/prompt_parts_test.cpp)
  target_include_directories(prompt_parts_test PRIVATE src)
  target_link_libraries(prompt_parts_test PRIVATE warnings)
  basalt_cpu_test(NAME prompt_parts_test COMMAND prompt_parts_test)
  # a round's windows over a batch's rows: each slot's share and the first-fit groups (header-only, CPU)
  add_executable(batch_plan_test tests/program/batch_plan_test.cpp)
  target_include_directories(batch_plan_test PRIVATE src)
  target_link_libraries(batch_plan_test PRIVATE warnings)
  basalt_cpu_test(NAME batch_plan_test COMMAND batch_plan_test)
  # which part of a held DeepSeek sequence the next prompt keeps (header-only, CPU)
  # the f32 -> bf16 rounding against ggml's, NaN included, over every f32 bit pattern (header-only, CPU)
  add_executable(bf16_bits_test tests/kernels/bf16_bits_test.cpp)
  target_include_directories(bf16_bits_test PRIVATE include)
  basalt_cpu_test(NAME bf16_bits_test COMMAND bf16_bits_test)
  # the tail skip's rule (rank, residency, drop) against hand-made and random windows (header-only, CPU)
  add_executable(route_lossy_test tests/kernels/route_lossy_test.cpp)
  target_include_directories(route_lossy_test PRIVATE include)
  basalt_cpu_test(NAME route_lossy_test COMMAND route_lossy_test)
endif()

add_library(kernels_cpu STATIC src/kernels/cpu/expert.cpp src/kernels/cpu/pool.cpp
            src/kernels/cpu/cpu_quota.cpp src/kernels/cpu/expert_layout.cpp src/kernels/cpu/q2_avx2.cpp
            src/kernels/cpu/q2_avx2_vnni.cpp)
target_include_directories(kernels_cpu PUBLIC include)
# only the kernel files get the AVX-512 / AVX2 flags; the pool and the probe run on any x86-64,
# and the engine picks the kernels at run time (cpu_avx512_ok).
# `-mavx512vnni` alone is not enough: VBMI is a separate feature bit.
set_source_files_properties(src/kernels/cpu/expert.cpp
                            PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags}")
set_source_files_properties(src/kernels/cpu/q2_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the same Q2_0 rows with the AVX-VNNI instructions, called only behind cpu_avxvnni_ok()
set_source_files_properties(src/kernels/cpu/q2_avx2_vnni.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avxvnni_flags}")
find_package(Threads REQUIRED)
target_link_libraries(kernels_cpu PUBLIC Threads::Threads platform)

# ---- ggml, extracted ----
#
# The parts of ggml (MIT, llama.cpp commit 3cf03257f219afbe7334045ff7c6a06ac68c627d) the engine uses, in
# src/kernels/cpu/ggml and src/prefill/mmq: the reference quantizers and the format table, the CPU row dot
# products, and the prompt path's MMQ kernels.  Nothing is fetched at build time.
add_library(ggml_quants STATIC src/kernels/cpu/ggml/ggml_quants.c)
target_include_directories(ggml_quants PUBLIC include PRIVATE src/kernels/cpu/ggml)
set_target_properties(ggml_quants PROPERTIES C_STANDARD 11 C_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
# The CPU dot products are built for AVX2 and for AVX-512, each with its own name for the table lookup;
# ggml_cpu_dispatch.cpp picks one at run time (cpu_avx512_ok / cpu_avx2_ok).
set(_ggml_dots_avx2_flags ${_isa_avx2_flags})
set(_ggml_dots_avx512_flags ${_isa_avx512_flags})
foreach(_isa avx2 avx512)
  add_library(ggml_dots_${_isa} OBJECT src/kernels/cpu/ggml/ggml_cpu_dots.c)
  target_include_directories(ggml_dots_${_isa} PRIVATE include src/kernels/cpu/ggml)
  target_compile_options(ggml_dots_${_isa} PRIVATE ${_ggml_dots_${_isa}_flags})
  target_compile_definitions(ggml_dots_${_isa} PRIVATE ggml_get_type_traits_cpu=ggml_get_type_traits_cpu_${_isa})
  set_target_properties(ggml_dots_${_isa} PROPERTIES C_STANDARD 11 C_EXTENSIONS ON POSITION_INDEPENDENT_CODE ON)
endforeach()
# part of kernels_cpu (the pool calls these and they call its CPU probe: one library, so GNU ld's
# one-pass static linking has no cycle to resolve)
target_sources(kernels_cpu PRIVATE src/kernels/cpu/native_expert.cpp src/kernels/cpu/iq_avx512.cpp
                                         src/kernels/cpu/nvfp4_avx512.cpp
                                         src/kernels/cpu/nvfp4_avx2.cpp src/kernels/cpu/nvfp4_avx2_vnni.cpp
                                         src/kernels/cpu/iq_avx2.cpp src/kernels/cpu/iq_avx2_vnni.cpp
                                         src/kernels/cpu/kq_avx2.cpp
                                         src/kernels/cpu/kq_avx1.cpp src/kernels/cpu/ggml/ggml_cpu_dispatch.cpp
                                         $<TARGET_OBJECTS:ggml_dots_avx2> $<TARGET_OBJECTS:ggml_dots_avx512>)
# the i-quant rows also use BMI2 (pdep/pext for the sign masks; cpu_avx512_ok requires it). MSVC compiles the
# intrinsics without a switch.
if(MSVC)
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags}")
else()
  set_source_files_properties(src/kernels/cpu/iq_avx512.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx512_flags};-mbmi2")
endif()
set_source_files_properties(src/kernels/cpu/nvfp4_avx512.cpp PROPERTIES COMPILE_OPTIONS
                            "${_isa_avx512_flags}")
set_source_files_properties(src/kernels/cpu/iq_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the same rows with the AVX-VNNI instructions and BMI2 (pdep for the grid indices), called only behind cpu_avxvnni_ok()
set_source_files_properties(src/kernels/cpu/iq_avx2_vnni.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avxvnni_flags}")
set_source_files_properties(src/kernels/cpu/nvfp4_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the same rows with AVX-VNNI-INT8 (`vpdpbssd`), called only behind cpu_avxvnni_int8_ok()
set_source_files_properties(src/kernels/cpu/nvfp4_avx2_vnni.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avxvnniint8_flags}")
set_source_files_properties(src/kernels/cpu/kq_avx2.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx2_flags}")
# the router lookahead's AVX1 dot (called behind cpu_avx1_ok)
set_source_files_properties(src/kernels/cpu/kq_avx1.cpp PROPERTIES COMPILE_OPTIONS "${_isa_avx1_flags}")
# MSVC gives the scalar F16C conversions no working code: its headers declare them nowhere, and with
# __F16C__ defined the compiler claims the names as its own, so a definition compiled under that macro does not
# build as written. The file below supplies the pair from the vector intrinsics, which are the same instruction,
# and it must be compiled without __F16C__ for its definitions to survive; the callers carry prototypes and
# #pragma function so the calls reach it. The file is empty for every other compiler, and it is not added to the
# library at all there.
if(MSVC)
  target_sources(kernels_cpu PRIVATE src/kernels/cpu/ggml/f16c_scalar_msvc.c)
  set_source_files_properties(src/kernels/cpu/ggml/f16c_scalar_msvc.c PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
target_link_libraries(kernels_cpu PUBLIC ggml_quants)
if(BUILD_TESTING)
  # the router lookahead's dot on each rung this CPU has (AVX2, the older CPUs' AVX1) against a double reference
  add_executable(router_dot_parity tests/kernels/cpu/router_dot_parity.cpp)
  target_link_libraries(router_dot_parity PRIVATE kernels_cpu)
  basalt_gpu_test(NAME router_dot_parity COMMAND router_dot_parity)
endif()
# MMQ (ggml-cuda) for the prompt path's experts: the kernel, one instance per format the packs use, and the
# activation quantizer; mmq_host.cu holds its few host symbols
set(_mmq_types q2_0 iq1_s iq1_m iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0 q5_0 q5_1 q2_k q3_k q4_k q5_k q6_k nvfp4)
# the Q2_0 pack's prompt experts on the engine's own int8 kernels (moe_fused.cu)
add_library(mmq STATIC src/prefill/moe_mmq.cu
            src/prefill/mmq/mmq_host.cu src/prefill/mmq/quantize.cu)
# q8_0: the drafter's matrices
foreach(_t IN LISTS _mmq_types)
  target_sources(mmq PRIVATE src/prefill/mmq/instances/mmq_${_t}.cu)
endforeach()
target_include_directories(mmq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -use_fast_math>)
# MSVC's COFF object holds 65536 sections; the instantiated MMQ formats exceed that in one TU without /bigobj.
if(MSVC)
  target_compile_options(mmq PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/bigobj>)
endif()
target_link_libraries(mmq PUBLIC ggml_quants ${_gpu_runtime_target})
# multi-GPU: a second card's share of a prompt chunk's routed experts, on the MMQ products
add_library(peer_share STATIC src/prefill/peer_share.cpp)
target_include_directories(peer_share PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include PRIVATE ${_gpu_include_directories})
target_link_libraries(peer_share PUBLIC mmq kernels ${_gpu_runtime_target})
# a prompt chunk's products against FP8 E4M3 weights on the FP8 tensor cores (W8A8)
add_library(prefill_fp8 STATIC src/prefill/fp8_gemm.cu)
set_target_properties(prefill_fp8 PROPERTIES CUDA_ARCHITECTURES 120a-real)   # the block-scaled mma
target_include_directories(prefill_fp8 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(prefill_fp8 PUBLIC core ${_gpu_runtime_target})
if(BUILD_TESTING)
  # the quantizer and product against CPU references, rows bitwise at any chunk length; --bench for TFLOP/s
  add_executable(fp8_gemm_test tests/kernels/fp8_gemm_test.cpp)
  target_link_libraries(fp8_gemm_test PRIVATE prefill_fp8 ${_gpu_runtime_target})
  basalt_gpu_test(NAME fp8_gemm_test COMMAND fp8_gemm_test)
  set_tests_properties(fp8_gemm_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # the producers that round their rows to E4M3 for an F8 output projection: bitwise quantize's
  add_executable(e4m3_producers_test tests/kernels/e4m3_producers_test.cpp)
  target_link_libraries(e4m3_producers_test PRIVATE prefill_fp8 kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME e4m3_producers_test COMMAND e4m3_producers_test)
  set_tests_properties(e4m3_producers_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
  # the prompt chunk's BF16 entry points against the FP32 kernels on the widened values
  add_executable(bf16_chunk_test tests/kernels/bf16_chunk_test.cpp)
  target_link_libraries(bf16_chunk_test PRIVATE prefill_fp8 prefill_fp4 kernels ${_gpu_runtime_target})
  basalt_gpu_test(NAME bf16_chunk_test COMMAND bf16_chunk_test)
  set_tests_properties(bf16_chunk_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
# a decode window's products against FP8 and NVFP4 matrices (the NVFP4 codes convert with sm_120a instructions)
add_library(window_gemm STATIC src/kernels/cuda/window_gemm.cu src/kernels/cuda/window_gemm_f8.cu
                               src/kernels/cuda/window_gemm_nvfp4.cu)
set_target_properties(window_gemm PROPERTIES CUDA_ARCHITECTURES 120a-real)
target_include_directories(window_gemm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(window_gemm PUBLIC core ${_gpu_runtime_target})
add_library(prefill_fp4 STATIC src/prefill/moe_fp4.cu src/prefill/fp4_dense.cu)
set_target_properties(prefill_fp4 PROPERTIES CUDA_ARCHITECTURES 120a-real)
target_include_directories(prefill_fp4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(prefill_fp4 PUBLIC mmq ${_gpu_runtime_target})
if(BUILD_TESTING)
  # a small chunk's experts on the CPU pool against MMQ and a double-precision reference
  add_executable(prefill_cpu_share_test tests/cuda/prefill_cpu_share_test.cpp)
  target_link_libraries(prefill_cpu_share_test PRIVATE mmq kernels_cpu ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_cpu_share_test COMMAND prefill_cpu_share_test)
  set_tests_properties(prefill_cpu_share_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
  # a second card's share of a chunk's experts against the main card's MMQ and a double-precision reference (two GPUs)
  add_executable(peer_share_parity tests/prefill/peer_share_parity.cu)
  target_link_libraries(peer_share_parity PRIVATE peer_share core ggml_quants ${_gpu_runtime_target})
  basalt_gpu_test(NAME peer_share_parity COMMAND peer_share_parity)
  set_tests_properties(peer_share_parity PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 600)
endif()
if(BUILD_TESTING)
  # NVFP4 experts through MMQ (tile loader, per-16-scale dot, scale tail in SwiGLU) against CPU references, with a
  # microbench line
  add_executable(prefill_mmq_nvfp4_test tests/cuda/prefill_mmq_nvfp4_test.cpp)
  target_link_libraries(prefill_mmq_nvfp4_test PRIVATE mmq ${_gpu_runtime_target})
  basalt_gpu_test(NAME prefill_mmq_nvfp4_test COMMAND prefill_mmq_nvfp4_test --no-bench)
  set_tests_properties(prefill_mmq_nvfp4_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 300)
endif()
if(BUILD_TESTING)
  # NVFP4 block decode against the E2M1/UE4M3 spec (exhaustive) and ggml's row dequantizer; with an experts.bin,
  # a real NVFP4 expert's blocks and scale tail
  add_executable(nvfp4_format_test tests/core/nvfp4_format_test.cpp)
  target_link_libraries(nvfp4_format_test PRIVATE artifact ggml_quants)
  basalt_cpu_test(NAME nvfp4_format_test COMMAND nvfp4_format_test)
  set_tests_properties(nvfp4_format_test PROPERTIES TIMEOUT 60)
endif()
if(BUILD_TESTING)
  add_executable(native_expert_parity tests/kernels/native_expert_parity.cpp)
  target_include_directories(native_expert_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(native_expert_parity PRIVATE kernels kernels_cpu artifact ${_gpu_runtime_target})
  add_executable(native_expert_bench tests/kernels/native_expert_bench.cpp)
  target_include_directories(native_expert_bench PRIVATE ${_gpu_include_directories})
  target_link_libraries(native_expert_bench PRIVATE kernels kernels_cpu artifact ${_gpu_runtime_target})
  # IQ1_S and the limited-SwiGLU expert pairs at the DeepSeek geometry, GPU and CPU tiers against a double reference
  add_executable(iq1s_parity tests/kernels/iq1s_parity.cpp)
  target_include_directories(iq1s_parity PRIVATE ${_gpu_include_directories} ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_compile_definitions(iq1s_parity PRIVATE BASALT_IQ1S_GPU)
  target_link_libraries(iq1s_parity PRIVATE kernels kernels_cpu ${_gpu_runtime_target})
  basalt_gpu_test(NAME iq1s_parity COMMAND iq1s_parity)
  set_tests_properties(iq1s_parity PROPERTIES TIMEOUT 300)
  basalt_cpu_test(NAME iq1s_parity_cpu COMMAND iq1s_parity --cpu)
  # native expert pairs on synthetic ggml-quantized weights at 2560/640: GPU and CPU experts against the float
  # reference, GPU dequantizers against to_float.
  foreach(_pair q4_K/q5_1 q4_K/q8_0 q5_K/q8_0 q6_K/q8_0 q8_0/q8_0 q4_K/q5_0 q5_0/q8_0 q2_K/q8_0 q3_K/q8_0)
    string(REPLACE "/" "_" _name "${_pair}")
    basalt_gpu_test(NAME native_expert_parity_${_name} COMMAND native_expert_parity --synthetic ${_pair})
    set_tests_properties(native_expert_parity_${_name} PROPERTIES TIMEOUT 120)
  endforeach()
  foreach(_pair q6_K/q8_0 q2_K/q8_0 q3_K/q8_0)
    string(REPLACE "/" "_" _name "${_pair}")
    basalt_cpu_test(NAME native_expert_parity_cpu_${_name} COMMAND native_expert_parity --cpu-synthetic ${_pair})
  endforeach()
  basalt_gpu_test(NAME native_expert_parity_q5_1_min COMMAND native_expert_parity --q5_1-min)
  # the AVX-512 Q8_K activation quantizer bit for bit against ggml's reference
  basalt_cpu_test(NAME native_expert_parity_q8k_quant COMMAND native_expert_parity --q8k-quant)
  # the BF16 token embedding gathered and dequantized bit for bit
  basalt_gpu_test(NAME native_expert_parity_bf16_embd COMMAND native_expert_parity --bf16-embd)
  # the Q6_K token embedding dequantized against ggml's reference
  basalt_gpu_test(NAME native_expert_parity_q6k_embd COMMAND native_expert_parity --q6k-embd)
  # the Q8_0 token embedding with its hot rows in VRAM, bit for bit against the mapped-only gather and ggml's reference
  basalt_gpu_test(NAME native_expert_parity_q8_embd_hot COMMAND native_expert_parity --q8-embd-hot)
  # NVFP4 on the CPU tier: AVX2/AVX-512 dots vs plain C, the expert with its tail scales vs a double reference
  basalt_gpu_test(NAME native_expert_parity_nvfp4 COMMAND native_expert_parity --synthetic nvfp4/nvfp4 nvfp4/nvfp4)
  set_tests_properties(native_expert_parity_nvfp4 PROPERTIES TIMEOUT 120)
endif()
# real-weight checks: test programs, not part of a release
if(BUILD_TESTING)
  add_executable(q5_projection_parity tests/kernels/q5_projection_parity.cpp)
  target_include_directories(q5_projection_parity PRIVATE ${_gpu_include_directories})
  target_link_libraries(q5_projection_parity PRIVATE kernels artifact ggml_quants ${_gpu_runtime_target})
endif()

if(BUILD_TESTING)
  # Both programs test a Linux-only surface: the cgroup quota files and the Linux affinity backend's headers.
  # Neither exists on Windows, so neither is built or run there.
  if(NOT WIN32)
    add_executable(cpu_quota_test tests/kernels/cpu/cpu_quota_test.cpp)
    target_link_libraries(cpu_quota_test PRIVATE kernels_cpu)
    basalt_cpu_test(NAME cpu_quota_test COMMAND cpu_quota_test)
    add_executable(pool_affinity_test tests/platform/affinity_linux_test.cpp)
    target_include_directories(pool_affinity_test PRIVATE src/platform)
    target_link_libraries(pool_affinity_test PRIVATE kernels_cpu)
    basalt_cpu_test(NAME pool_affinity_test COMMAND pool_affinity_test)
  endif()
  # one expert applied to several tokens, bitwise vs the single-token kernel (synthetic, CPU only).
  # the device dequantizers vs the artifact's CPU dequantizers on the real shards (not a ctest: needs
  # the full model; run it by hand)
  add_executable(dequant_bf16_test tests/kernels/dequant_bf16_test.cpp)
  target_link_libraries(dequant_bf16_test PRIVATE kernels artifact ${_gpu_runtime_target})

  # the CPU expert kernels on each instruction set side by side (timing; the parity checks are native_expert_parity)
  add_executable(avx_bench tests/kernels/cpu/avx_bench.cpp)
  target_link_libraries(avx_bench PRIVATE kernels_cpu)

  # needs the AVX-512 VNNI and VBMI instruction sets, which not every host has

  add_executable(nvfp4_cpu_rows_test tests/kernels/cpu/nvfp4_cpu_rows_test.cpp)
  target_link_libraries(nvfp4_cpu_rows_test PRIVATE kernels_cpu)
  basalt_cpu_test(NAME nvfp4_cpu_rows_test COMMAND nvfp4_cpu_rows_test)
  set_tests_properties(nvfp4_cpu_rows_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "cpu;avx2")

  # Both tests run on a synthetic expert fixture by default; --file experts.bin (the pack's experts section) checks real blobs.

  # tiny batches with sleeping workers, as on a big-VRAM card; a hang fails it
  add_executable(pool_stress tests/kernels/cpu/pool_stress.cpp)
  target_link_libraries(pool_stress PRIVATE kernels_cpu)
  basalt_cpu_test(NAME pool_stress COMMAND pool_stress 20)
  # the multi-token native path against a serial reference, bitwise, at several worker counts
  add_executable(pool_native_test tests/kernels/cpu/pool_native_test.cpp)
  target_link_libraries(pool_native_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME pool_native_test COMMAND pool_native_test)
  # the AVX2 i-quant rows: the madd build against the AVX-VNNI build (bitwise) and a double reference
  add_executable(iq_avx2_test tests/kernels/cpu/iq_avx2_test.cpp)
  target_link_libraries(iq_avx2_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME iq_avx2_test COMMAND iq_avx2_test)
  # the AVX-512 i-quant rows: frozen output hashes, token-count independence and a double reference
  add_executable(iq_avx512_test tests/kernels/cpu/iq_avx512_test.cpp)
  target_link_libraries(iq_avx512_test PRIVATE kernels_cpu platform)
  basalt_cpu_test(NAME iq_avx512_test COMMAND iq_avx512_test)
  set_property(TEST iq_avx512_test APPEND PROPERTY LABELS avx512)

  # DirectFile has a Linux backend and a Windows backend, and this test covers those two; macOS is not a platform
  # the engine builds for.
  if((UNIX AND NOT APPLE) OR WIN32)
    add_executable(direct_file_async_test tests/platform/direct_file_async.cpp)
    target_include_directories(direct_file_async_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(direct_file_async_test PRIVATE BASALT_TEST_TMPDIR="${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(direct_file_async_test PRIVATE platform Threads::Threads)
    basalt_embed_manifest(direct_file_async_test)
    basalt_cpu_test(NAME direct_file_async_test COMMAND direct_file_async_test)
  endif()
endif()
