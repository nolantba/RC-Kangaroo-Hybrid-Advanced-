// GPU Monitoring and Thermal Management System
// Provides per-GPU statistics, thermal throttling, and load balancing

#ifndef GPU_MONITOR_H
#define GPU_MONITOR_H

#include "defs.h"

// Optional NVML support - requires nvidia-ml-dev package
#ifdef USE_NVML
#include <nvml.h>
#endif

// Per-GPU statistics
struct GPUStats {
    int gpu_id;
    double speed_mkeys;       // MKeys/s for this GPU
    unsigned int temp_c;      // Temperature in Celsius
    unsigned int power_mw;    // Power draw in milliwatts
    unsigned int util_pct;    // GPU utilization percentage
    unsigned int mem_used_mb; // Memory used in MB
    int pci_bus;              // PCI bus ID
    bool throttling;          // Is GPU thermal throttling?
    u64 operations;           // Total operations completed
};

// System-wide monitoring
struct SystemStats {
    GPUStats gpu_stats[MAX_GPU_CNT];
    int gpu_count;
    double total_gpu_speed;
    double cpu_speed_mkeys;
    unsigned int avg_temp_c;
    unsigned int total_power_w;

    // K-factor tracking
    double current_k_factor;
    double expected_ops;
    double actual_ops;

    // DP health
    u64 dp_count;
    u64 dp_expected;
    double dp_rate_per_sec;
    u32 dp_buffer_used;
    u32 dp_buffer_total;

    // Timing
    u64 start_time_ms;
    u64 elapsed_ms;
    u64 eta_ms;
};

// Thermal management policies
enum ThermalPolicy {
    THERMAL_AGGRESSIVE,  // Max performance, temps up to 85°C
    THERMAL_BALANCED,    // Balance performance/temps, keep < 80°C
    THERMAL_QUIET        // Low noise, keep < 75°C
};

// GPU Monitor class
class GpuMonitor {
public:
    GpuMonitor();
    ~GpuMonitor();

    // Initialization
    bool Initialize(int gpu_count);
    void Shutdown();

    // Monitoring
    bool UpdateStats(int gpu_id);
    void UpdateAllGPUs();
    GPUStats GetGPUStats(int gpu_id);
    SystemStats GetSystemStats();
    void SetSystemStats(const SystemStats& stats);  // Update system stats from outside

    // Thermal management
    void SetThermalPolicy(ThermalPolicy policy);
    bool ApplyThermalLimits();  // Returns true if throttling applied
    void RestorePowerLimits();

    // Load balancing
    void CalculateOptimalDistribution(int total_kangaroos, int* per_gpu_kangaroos);
    double GetGPUEfficiency(int gpu_id);  // Operations per watt

    // Display
    void PrintDetailedStats();
    void PrintCompactStats();

private:
#ifdef USE_NVML
    nvmlDevice_t nvml_devices[MAX_GPU_CNT];
#endif
    GPUStats gpu_stats[MAX_GPU_CNT];
    SystemStats sys_stats;
    ThermalPolicy thermal_policy;
    bool initialized;

    // Thermal thresholds
    unsigned int temp_warning;   // Start monitoring
    unsigned int temp_throttle;  // Reduce power
    unsigned int temp_critical;  // Emergency reduction

    // Power limits (in watts)
    unsigned int power_limit_default[MAX_GPU_CNT];
    unsigned int power_limit_throttle[MAX_GPU_CNT];

    // Performance tracking for load balancing
    double gpu_performance_history[MAX_GPU_CNT][60]; // Last 60 samples
    int performance_sample_idx;

    // Helper functions
    bool InitNVML();
    void UpdateThermalThresholds();
    double CalculateMovingAverage(int gpu_id, int samples);
};

// Global monitor instance (declared in GpuMonitor.cpp)
extern GpuMonitor* g_gpu_monitor;

#endif // GPU_MONITOR_H
