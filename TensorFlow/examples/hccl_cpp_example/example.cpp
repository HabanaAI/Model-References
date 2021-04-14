/******************************************************************************
 * Copyright (C) 2021 Habana Labs, Ltd. an Intel Company
 * All Rights Reserved.
 *
 * Unauthorized copying of this file or any element(s) within it, via any medium
 * is strictly prohibited.
 * This file contains Habana Labs, Ltd. proprietary and confidential information
 * and is subject to the confidentiality and license agreements under which it
 * was provided.
 *
 *******************************************************************************
 */

// C++ Standard Libraries
#include <iostream>
#include <exception>

// Open MPI (v4.0.2)
#include "mpi.h"

// HCCL :: Habana Collective Communications Library
#include <hccl/hccl.h>

// ------------------------------------------------------------------------------------------------

#define CHECK_MPI_STATUS(x)                                                             \
  {                                                                                     \
    const auto _res = (x);                                                              \
    if (_res != MPI_SUCCESS)                                                            \
      throw std::runtime_error{"In function " + std::string{__FUNCTION__} +             \
                               "(): " #x " failed with code: " + std::to_string(_res)}; \
  }

#define CHECK_HCCL_STATUS(x)                                                      \
  {                                                                               \
    const auto _res = (x);                                                        \
    if (_res != hcclSuccess)                                                      \
      throw std::runtime_error{"In function " + std::string{__FUNCTION__} +       \
                               "(): " #x " failed: " + hcclGetErrorString(_res)}; \
  }

std::ostream& log() { return std::cout; }

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& data) {
  out << '[';
  if (!data.empty()) {
    out << data[0];
    for (size_t i = 1; i < data.size(); ++i) {
      out << " " << data[i];
    }
  }
  return out << ']';
}

// ------------------------------------------------------------------------------------------------

int main(int argc, char* argv[]) {
  try {
    log() << "Running HCCL Example :: A simple program demonstrating HCCL usage from C++" << std::endl;

    // Initialize the Open MPI execution context.
    CHECK_MPI_STATUS(MPI_Init(&argc, &argv));

    // Get MPI rank and size.
    int mpi_rank{};
    CHECK_MPI_STATUS(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));

    int mpi_size{};
    CHECK_MPI_STATUS(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

    if (mpi_size < 2) {
      throw std::runtime_error{"There is only one MPI rank. Run this program using 'mpirun' or 'mpiexec'"};
    }

    // Initialize HCCL.
    // As TensorFlow is not used here, HCCL will create a new HPU device and use its own HPU allocator.
    // This step can be omitted only if there is an other agent who already initialized HCCL.
    // Habana's version of Horovod with HCCL backend does that by instantiating hccl::gpu_context class.
    // hcclDeviceInit() is a C function which internally creates a new instance of hccl::gpu_context class.
    // If you prefer C++ API, you can use it directly either by creating an new instance of this class,
    // or accessing an existing instance using gpu_context::instance().
    //
    int device_handle{};
    const int device_cardinal_id = 0;  // There may be only one HPU device per process, so it is always 0.

    CHECK_HCCL_STATUS(hcclDeviceInit(&device_handle, device_cardinal_id));

    // Generate Unique Id on rank 0 and propagate it to other ranks using Open MPI.
    //
    hcclUniqueId unique_id{};
    constexpr int master_mpi_rank = 0;

    if (mpi_rank == master_mpi_rank) {
      CHECK_HCCL_STATUS(hcclGetUniqueId(&unique_id));
    }

    CHECK_MPI_STATUS(MPI_Bcast(&unique_id, sizeof(unique_id), MPI_BYTE, master_mpi_rank, MPI_COMM_WORLD));

    // Create a new HCCL communicator.
    hcclComm_t hccl_comm{};
    CHECK_HCCL_STATUS(hcclCommInitRank(&hccl_comm, mpi_size, unique_id, mpi_rank));

    // Create a new stream handle. There is no need to destroy it any way.
    hcclStream_t stream{};
    CHECK_HCCL_STATUS(hcclStreamCreate(&stream));

    // Allocate some buffer on the HPU device.
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
    auto input_host_data = std::vector<float>{1.5f, -2.0f, -0.5f, 3.0f};
    const size_t data_size = 4 * sizeof(float);

    void* dev_ptr{};
    CHECK_HCCL_STATUS(hcclMalloc(&dev_ptr, data_size));
    CHECK_HCCL_STATUS(hcclMemcpy(dev_ptr, &input_host_data.front(), data_size, hcclMemcpyHostToDevice, stream));

    // There is no need to call hcclWaitForCompletion(stream).
    // A synchronization on dev_ptr device address will be automatic.

    // Perform an All-Reduce operation on the device buffer.
    CHECK_HCCL_STATUS(hcclAllReduce(dev_ptr, dev_ptr, data_size, hcclFloat32, hcclSum, hccl_comm, stream));

    // There is no need to call hcclWaitForCompletion(stream).
    // A synchronization on dev_ptr device address will be automatic.

    // Copy the data back to the host memory.
    auto output_host_data = std::vector<float>(4);
    CHECK_HCCL_STATUS(hcclMemcpy(&output_host_data.front(), dev_ptr, data_size, hcclMemcpyDeviceToHost, stream));

    // There is no need to call hcclWaitForCompletion(stream), as device->host memcopy is blocking.

    // Check if the data has been reduced correctly.
    bool is_ok = true;
    for (size_t i = 0; i < input_host_data.size(); ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
      if (std::abs(output_host_data[i] - static_cast<float>(mpi_size) * input_host_data[i]) > 1e-10f) {
        is_ok = false;
      }
    }

    log() << "Buffer " << input_host_data << " reduced to " << output_host_data << " which is "
          << (is_ok ? "fine." : "bad.") << std::endl;

    // Free up resources.
    CHECK_HCCL_STATUS(hcclFree(dev_ptr));

    // Destroy a HCCL communicator.
    CHECK_HCCL_STATUS(hcclCommDestroy(hccl_comm));

    // Clean up HCCL.
    CHECK_HCCL_STATUS(hcclDeviceFree(device_cardinal_id));

    // Clean up Open MPI.
    CHECK_MPI_STATUS(MPI_Finalize());

  } catch (const std::exception& ex) {
    log() << "error: " << ex.what() << std::endl;
    return -1;
  }

  return 0;
}

// ------------------------------------------------------------------------------------------------
