#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <chrono>
#include <cstddef>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct TransitionInput {
    float dx;
    float dy;
    float consumption;
    float velocity;
};

struct TransitionOutput {
    float distance;
    float energy;
    float travel_time;
};

constexpr char kMetalSource[] = R"metal(
#include <metal_stdlib>
using namespace metal;

struct TransitionInput {
    float dx;
    float dy;
    float consumption;
    float velocity;
};

struct TransitionOutput {
    float distance;
    float energy;
    float travel_time;
};

kernel void transition_kernel(
    const device TransitionInput* inputs [[buffer(0)]],
    device TransitionOutput* outputs [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const float dx = inputs[index].dx;
    const float dy = inputs[index].dy;
    const float distance = sqrt(dx * dx + dy * dy);
    outputs[index].distance = distance;
    outputs[index].energy = distance * inputs[index].consumption;
    outputs[index].travel_time = distance / inputs[index].velocity;
}
)metal";

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    bool initialized = false;
    bool available = false;
    bool initialization_reported = false;
    std::string reason;
    double initialization_seconds = 0.0;

    bool ensure() {
        if (initialized) {
            return available;
        }
        const auto started = std::chrono::steady_clock::now();
        initialized = true;
        device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            reason = "MTLCreateSystemDefaultDevice returned nil";
            initialization_seconds = elapsed_seconds(started);
            return false;
        }
        queue = [device newCommandQueue];
        if (queue == nil) {
            reason = "Metal command queue creation failed";
            initialization_seconds = elapsed_seconds(started);
            return false;
        }
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:kMetalSource];
        id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
        if (library == nil) {
            reason = error == nil
                ? "Metal shader compilation returned nil"
                : std::string([[error localizedDescription] UTF8String]);
            initialization_seconds = elapsed_seconds(started);
            return false;
        }
        id<MTLFunction> function = [library newFunctionWithName:@"transition_kernel"];
        if (function == nil) {
            reason = "compiled Metal library has no transition_kernel function";
            initialization_seconds = elapsed_seconds(started);
            return false;
        }
        pipeline = [device newComputePipelineStateWithFunction:function error:&error];
        if (pipeline == nil) {
            reason = error == nil
                ? "Metal compute pipeline creation returned nil"
                : std::string([[error localizedDescription] UTF8String]);
            initialization_seconds = elapsed_seconds(started);
            return false;
        }
        available = true;
        initialization_seconds = elapsed_seconds(started);
        return true;
    }

    static double elapsed_seconds(
        const std::chrono::steady_clock::time_point& started) {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
    }
};

MetalState& state() {
    static MetalState value;
    static std::once_flag once;
    std::call_once(once, []() {
        // Construction is intentionally lazy and thread-safe. The pilot is
        // single-threaded, but this keeps the native seam deterministic if a
        // capability probe is made from a test harness.
    });
    return value;
}

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

void require_vector(const FloatArray& array, const char* name, std::size_t expected) {
    const auto info = array.request();
    if (info.ndim != 1 || static_cast<std::size_t>(info.shape[0]) != expected) {
        throw std::invalid_argument(std::string("Metal input ") + name + " must be a 1-D vector");
    }
}

py::dict metal_backend_info() {
    MetalState& metal = state();
    const bool available = metal.ensure();
    py::dict result;
    result["available"] = available;
    result["reason"] = metal.reason;
    result["device_name"] = available
        ? std::string([[metal.device name] UTF8String])
        : std::string();
    result["initialization_seconds"] = metal.initialization_reported
        ? 0.0
        : metal.initialization_seconds;
    metal.initialization_reported = true;
    result["shader"] = "transition_kernel";
    return result;
}

py::dict metal_transition_batch(
    const FloatArray& dx,
    const FloatArray& dy,
    const FloatArray& consumption,
    const FloatArray& velocity) {
    MetalState& metal = state();
    if (!metal.ensure()) {
        throw std::runtime_error("Metal backend unavailable: " + metal.reason);
    }
    const std::size_t count = static_cast<std::size_t>(dx.request().shape[0]);
    require_vector(dy, "dy", count);
    require_vector(consumption, "consumption", count);
    require_vector(velocity, "velocity", count);
    py::array_t<float> output(
        py::array::ShapeContainer{
            static_cast<py::ssize_t>(count),
            static_cast<py::ssize_t>(3),
        });
    if (count == 0) {
        py::dict empty;
        empty["outputs"] = output;
        empty["kernel_seconds"] = 0.0;
        empty["transfer_seconds"] = 0.0;
        empty["initialization_seconds"] = 0.0;
        return empty;
    }

    const auto transfer_started = std::chrono::steady_clock::now();
    std::vector<TransitionInput> inputs(count);
    const float* dx_ptr = static_cast<const float*>(dx.request().ptr);
    const float* dy_ptr = static_cast<const float*>(dy.request().ptr);
    const float* consumption_ptr = static_cast<const float*>(consumption.request().ptr);
    const float* velocity_ptr = static_cast<const float*>(velocity.request().ptr);
    for (std::size_t index = 0; index < count; ++index) {
        inputs[index] = {
            dx_ptr[index],
            dy_ptr[index],
            consumption_ptr[index],
            velocity_ptr[index],
        };
    }
    id<MTLBuffer> input_buffer = [metal.device
        newBufferWithBytes:inputs.data()
        length:inputs.size() * sizeof(TransitionInput)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer = [metal.device
        newBufferWithLength:count * sizeof(TransitionOutput)
        options:MTLResourceStorageModeShared];
    if (input_buffer == nil || output_buffer == nil) {
        throw std::runtime_error("Metal buffer allocation failed");
    }

    id<MTLCommandBuffer> command_buffer = [metal.queue commandBuffer];
    if (command_buffer == nil) {
        throw std::runtime_error("Metal command buffer allocation failed");
    }
    id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
    if (encoder == nil) {
        throw std::runtime_error("Metal compute encoder allocation failed");
    }
    [encoder setComputePipelineState:metal.pipeline];
    [encoder setBuffer:input_buffer offset:0 atIndex:0];
    [encoder setBuffer:output_buffer offset:0 atIndex:1];
    const NSUInteger width = std::max<NSUInteger>(1, metal.pipeline.threadExecutionWidth);
    const MTLSize grid = MTLSizeMake(count, 1, 1);
    const MTLSize group = MTLSizeMake(std::min<NSUInteger>(width, count), 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:group];
    [encoder endEncoding];
    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if ([command_buffer status] != MTLCommandBufferStatusCompleted) {
        NSString* message = [command_buffer error] == nil
            ? @"Metal command buffer did not complete"
            : [[[command_buffer error] localizedDescription] copy];
        throw std::runtime_error([message UTF8String]);
    }

    std::vector<TransitionOutput> values(count);
    std::memcpy(
        values.data(),
        [output_buffer contents],
        values.size() * sizeof(TransitionOutput));
    const double transfer_seconds = MetalState::elapsed_seconds(transfer_started);
    const double gpu_start = [command_buffer GPUStartTime];
    const double gpu_end = [command_buffer GPUEndTime];
    const double kernel_seconds = gpu_end >= gpu_start && gpu_start > 0.0
        ? gpu_end - gpu_start
        : transfer_seconds;
    float* output_ptr = static_cast<float*>(output.request().ptr);
    for (std::size_t index = 0; index < count; ++index) {
        output_ptr[index * 3] = values[index].distance;
        output_ptr[index * 3 + 1] = values[index].energy;
        output_ptr[index * 3 + 2] = values[index].travel_time;
    }
    py::dict result;
    result["outputs"] = output;
    result["kernel_seconds"] = kernel_seconds;
    result["transfer_seconds"] = transfer_seconds;
    result["initialization_seconds"] = 0.0;
    return result;
}

}  // namespace

void bind_metal(py::module_& module) {
    module.def("metal_backend_info", &metal_backend_info);
    module.def(
        "metal_transition_batch",
        &metal_transition_batch,
        py::arg("dx"),
        py::arg("dy"),
        py::arg("consumption"),
        py::arg("velocity"));
}
