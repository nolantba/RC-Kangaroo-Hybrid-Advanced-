// GPU Monitoring and Thermal Management Implementation

#include "GpuMonitor.h"
#include "utils.h"
#include <cstring>
#include <cmath>

GpuMonitor* g_gpu_monitor = nullptr;

GpuMonitor::GpuMonitor() {
    initialized = false;
    thermal_policy = THERMAL_BALANCED;
    performance_sample_idx = 0;
    memset(&sys_stats, 0, sizeof(sys_stats));
    memset(gpu_stats, 0, sizeof(gpu_stats));
    memset(gpu_performance_history, 0, sizeof(gpu_performance_history));
    memset(power_limit_default, 0, sizeof(power_limit_default));
    memset(power_limit_throttle, 0, sizeof(power_limit_throttle));
}

GpuMonitor::~GpuMonitor() {
    Shutdown();
}

bool GpuMonitor::Initialize(int gpu_count) {
    if (gpu_count <= 0 || gpu_count > MAX_GPU_CNT) {
        printf("GPU Monitor: Invalid GPU count: %d\n", gpu_count);
        return false;
    }

    sys_stats.gpu_count = gpu_count;

#ifdef USE_NVML
    // Initialize NVML
    if (!InitNVML()) {
        printf("GPU Monitor: Failed to initialize NVML\n");
        return false;
    }

    // Get NVML device handles
    for (int i = 0; i < gpu_count; i++) {
        nvmlReturn_t result = nvmlDeviceGetHandleByIndex(i, &nvml_devices[i]);
        if (result != NVML_SUCCESS) {
            printf("GPU Monitor: Failed to get device handle for GPU %d: %s\n",
                   i, nvmlErrorString(result));
            return false;
        }

        gpu_stats[i].gpu_id = i;

        // Get default power limit
        unsigned int power_limit;
        result = nvmlDeviceGetPowerManagementLimit(nvml_devices[i], &power_limit);
        if (result == NVML_SUCCESS) {
            power_limit_default[i] = power_limit / 1000; // mW to W
            power_limit_throttle[i] = (unsigned int)(power_limit_default[i] * 0.85); // 85% for throttling
        } else {
            power_limit_default[i] = 170; // RTX 3060 default
            power_limit_throttle[i] = 145;
        }

        // Get PCI bus ID
        nvmlPciInfo_t pci_info;
        result = nvmlDeviceGetPciInfo(nvml_devices[i], &pci_info);
        if (result == NVML_SUCCESS) {
            gpu_stats[i].pci_bus = pci_info.bus;
        }
    }

    UpdateThermalThresholds();
    initialized = true;

    printf("GPU Monitor: Initialized for %d GPUs\n", gpu_count);
    printf("Thermal Policy: ");
    switch (thermal_policy) {
        case THERMAL_AGGRESSIVE: printf("AGGRESSIVE (< 85°C)\n"); break;
        case THERMAL_BALANCED:   printf("BALANCED (< 80°C)\n"); break;
        case THERMAL_QUIET:      printf("QUIET (< 75°C)\n"); break;
    }

    return true;
#else
    printf("GPU Monitor: NVML support not compiled (USE_NVML not defined)\n");
    printf("GPU Monitor: Running with limited functionality\n");

    // Initialize basic stats without NVML
    for (int i = 0; i < gpu_count; i++) {
        gpu_stats[i].gpu_id = i;
        gpu_stats[i].pci_bus = i;
        power_limit_default[i] = 170;
        power_limit_throttle[i] = 145;
    }

    UpdateThermalThresholds();
    initialized = true;
    return true;
#endif
}

void GpuMonitor::Shutdown() {
    if (initialized) {
        RestorePowerLimits();
#ifdef USE_NVML
        nvmlShutdown();
#endif
        initialized = false;
    }
}

bool GpuMonitor::InitNVML() {
#ifdef USE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        printf("Failed to initialize NVML: %s\n", nvmlErrorString(result));
        return false;
    }
    return true;
#else
    return false;
#endif
}

void GpuMonitor::UpdateThermalThresholds() {
    switch (thermal_policy) {
        case THERMAL_AGGRESSIVE:
            temp_warning = 80;
            temp_throttle = 83;
            temp_critical = 85;
            break;
        case THERMAL_BALANCED:
            temp_warning = 75;
            temp_throttle = 78;
            temp_critical = 80;
            break;
        case THERMAL_QUIET:
            temp_warning = 70;
            temp_throttle = 73;
            temp_critical = 75;
            break;
    }
}

bool GpuMonitor::UpdateStats(int gpu_id) {
    if (!initialized || gpu_id < 0 || gpu_id >= sys_stats.gpu_count) {
        return false;
    }

#ifdef USE_NVML
    GPUStats* stats = &gpu_stats[gpu_id];
    nvmlDevice_t device = nvml_devices[gpu_id];

    // Temperature
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &stats->temp_c);

    // Power
    nvmlDeviceGetPowerUsage(device, &stats->power_mw);

    // Utilization
    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
        stats->util_pct = util.gpu;
    }

    // Memory
    nvmlMemory_t mem_info;
    if (nvmlDeviceGetMemoryInfo(device, &mem_info) == NVML_SUCCESS) {
        stats->mem_used_mb = (unsigned int)(mem_info.used / (1024 * 1024));
    }

    // Check for throttling
    stats->throttling = (stats->temp_c >= temp_throttle);

    return true;
#else
    // Without NVML, can't update hardware stats
    return false;
#endif
}

void GpuMonitor::UpdateAllGPUs() {
    sys_stats.total_gpu_speed = 0;
    sys_stats.avg_temp_c = 0;
    sys_stats.total_power_w = 0;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        UpdateStats(i);
        // CRITICAL: Copy speed from sys_stats to gpu_stats for display!
        gpu_stats[i].speed_mkeys = sys_stats.gpu_stats[i].speed_mkeys;
        sys_stats.total_gpu_speed += gpu_stats[i].speed_mkeys / 1000.0; // Convert to GKeys/s
        sys_stats.avg_temp_c += gpu_stats[i].temp_c;
        sys_stats.total_power_w += gpu_stats[i].power_mw / 1000;
    }

    if (sys_stats.gpu_count > 0) {
        sys_stats.avg_temp_c /= sys_stats.gpu_count;
    }

    // Update elapsed time and ETA
    sys_stats.elapsed_ms = GetTickCount64() - sys_stats.start_time_ms;

    // Calculate ETA based on current K-factor
    if (sys_stats.actual_ops > 0 && sys_stats.expected_ops > 0) {
        sys_stats.current_k_factor = sys_stats.actual_ops / sys_stats.expected_ops;
        double total_ops_needed = sys_stats.expected_ops * 1.15; // SOTA theoretical
        double ops_remaining = total_ops_needed - sys_stats.actual_ops;
        double current_speed = (sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0) * 1e9; // ops/sec
        if (current_speed > 0) {
            sys_stats.eta_ms = (u64)((ops_remaining / current_speed) * 1000.0);
        }
    }
}

GPUStats GpuMonitor::GetGPUStats(int gpu_id) {
    if (gpu_id >= 0 && gpu_id < sys_stats.gpu_count) {
        return gpu_stats[gpu_id];
    }
    GPUStats empty = {0};
    return empty;
}

SystemStats GpuMonitor::GetSystemStats() {
    return sys_stats;
}

void GpuMonitor::SetSystemStats(const SystemStats& stats) {
    sys_stats = stats;
}

void GpuMonitor::SetThermalPolicy(ThermalPolicy policy) {
    thermal_policy = policy;
    UpdateThermalThresholds();
    printf("GPU Monitor: Thermal policy changed to ");
    switch (policy) {
        case THERMAL_AGGRESSIVE: printf("AGGRESSIVE\n"); break;
        case THERMAL_BALANCED:   printf("BALANCED\n"); break;
        case THERMAL_QUIET:      printf("QUIET\n"); break;
    }
}

bool GpuMonitor::ApplyThermalLimits() {
#ifdef USE_NVML
    bool throttled = false;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        unsigned int temp = gpu_stats[i].temp_c;
        nvmlDevice_t device = nvml_devices[i];

        if (temp >= temp_critical) {
            // Critical: Reduce to 85% power
            unsigned int limit = power_limit_throttle[i] * 1000; // W to mW
            if (nvmlDeviceSetPowerManagementLimit(device, limit) == NVML_SUCCESS) {
                gpu_stats[i].throttling = true;
                throttled = true;
                if (temp >= temp_critical + 2) {
                    printf("⚠️  GPU %d CRITICAL TEMP: %u°C → Reduced to %uW\n",
                           i, temp, power_limit_throttle[i]);
                }
            }
        } else if (temp >= temp_throttle) {
            // Throttle: Reduce to 90% power
            unsigned int limit = (unsigned int)(power_limit_default[i] * 0.90) * 1000;
            if (nvmlDeviceSetPowerManagementLimit(device, limit) == NVML_SUCCESS) {
                gpu_stats[i].throttling = true;
                throttled = true;
            }
        } else if (temp < temp_warning && gpu_stats[i].throttling) {
            // Restore full power
            unsigned int limit = power_limit_default[i] * 1000;
            nvmlDeviceSetPowerManagementLimit(device, limit);
            gpu_stats[i].throttling = false;
        }
    }

    return throttled;
#else
    return false;
#endif
}

void GpuMonitor::RestorePowerLimits() {
#ifdef USE_NVML
    for (int i = 0; i < sys_stats.gpu_count; i++) {
        unsigned int limit = power_limit_default[i] * 1000;
        nvmlDeviceSetPowerManagementLimit(nvml_devices[i], limit);
        gpu_stats[i].throttling = false;
    }
#endif
}

void GpuMonitor::CalculateOptimalDistribution(int total_kangaroos, int* per_gpu_kangaroos) {
    // Calculate efficiency scores for each GPU
    double efficiency[MAX_GPU_CNT] = {0};
    double total_efficiency = 0;

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        // Efficiency = (speed / power) * thermal_factor
        double thermal_factor = 1.0;
        if (gpu_stats[i].temp_c >= temp_throttle) {
            thermal_factor = 0.7; // Penalize hot GPUs
        } else if (gpu_stats[i].temp_c >= temp_warning) {
            thermal_factor = 0.85;
        }

        if (gpu_stats[i].power_mw > 0) {
            efficiency[i] = (gpu_stats[i].speed_mkeys / (gpu_stats[i].power_mw / 1000.0)) * thermal_factor;
        } else {
            efficiency[i] = 1.0; // Equal if no power data
        }
        total_efficiency += efficiency[i];
    }

    // Distribute kangaroos proportionally to efficiency
    if (total_efficiency > 0) {
        int distributed = 0;
        for (int i = 0; i < sys_stats.gpu_count; i++) {
            if (i == sys_stats.gpu_count - 1) {
                // Last GPU gets remainder to ensure total is exact
                per_gpu_kangaroos[i] = total_kangaroos - distributed;
            } else {
                per_gpu_kangaroos[i] = (int)(total_kangaroos * (efficiency[i] / total_efficiency));
                distributed += per_gpu_kangaroos[i];
            }
        }
    } else {
        // Fallback: Equal distribution
        int per_gpu = total_kangaroos / sys_stats.gpu_count;
        for (int i = 0; i < sys_stats.gpu_count; i++) {
            per_gpu_kangaroos[i] = per_gpu;
        }
    }
}

double GpuMonitor::GetGPUEfficiency(int gpu_id) {
    if (gpu_id < 0 || gpu_id >= sys_stats.gpu_count) return 0;
    if (gpu_stats[gpu_id].power_mw == 0) return 0;
    return gpu_stats[gpu_id].speed_mkeys / (gpu_stats[gpu_id].power_mw / 1000.0);
}

void GpuMonitor::PrintDetailedStats() {
    printf("\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
    printf("GPU Performance Monitor\n");
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    for (int i = 0; i < sys_stats.gpu_count; i++) {
        GPUStats* s = &gpu_stats[i];
        printf("GPU %d: %.2f GK/s │ %3u°C │ %3uW │ %3u%% util │ PCI %3d",
               i, s->speed_mkeys / 1000.0, s->temp_c, s->power_mw / 1000,
               s->util_pct, s->pci_bus);

        if (s->throttling) {
            printf(" ⚠️ THROTTLING");
        } else if (s->temp_c >= temp_warning) {
            printf(" 🌡️ WARM");
        }
        printf("\n");
    }

    printf("CPU:   %.1f MK/s\n", sys_stats.cpu_speed_mkeys);
    printf("\nTotal: %.2f GK/s │ Avg Temp: %u°C │ Power: %uW\n",
           sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0,
           sys_stats.avg_temp_c, sys_stats.total_power_w);

    printf("\nK-Factor: %.3f ", sys_stats.current_k_factor);
    if (sys_stats.current_k_factor < 1.0) {
        printf("✓ (ahead of schedule)");
    } else if (sys_stats.current_k_factor < 1.15) {
        printf("✓ (on track)");
    } else if (sys_stats.current_k_factor < 1.3) {
        printf("⚠️ (slightly slow)");
    } else {
        printf("❌ (check for issues)");
    }
    printf("\n");

    printf("DPs: %llu / %llu (%.1f%%) │ Buffer: %u / %u (%.1f%%)\n",
           (unsigned long long)sys_stats.dp_count,
           (unsigned long long)sys_stats.dp_expected,
           (double)sys_stats.dp_count / sys_stats.dp_expected * 100.0,
           sys_stats.dp_buffer_used, sys_stats.dp_buffer_total,
           (double)sys_stats.dp_buffer_used / sys_stats.dp_buffer_total * 100.0);

    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}

void GpuMonitor::PrintCompactStats() {
    // Compact one-line status for regular updates
    printf("GPUs: ");
    for (int i = 0; i < sys_stats.gpu_count; i++) {
        printf("%d:%.1fGK/s,%u°C ", i, gpu_stats[i].speed_mkeys / 1000.0, gpu_stats[i].temp_c);
        if (gpu_stats[i].throttling) printf("⚠️ ");
    }
    printf("│ Total: %.2f GK/s │ K: %.3f",
           sys_stats.total_gpu_speed + sys_stats.cpu_speed_mkeys / 1000.0,
           sys_stats.current_k_factor);
}

double GpuMonitor::CalculateMovingAverage(int gpu_id, int samples) {
    if (gpu_id < 0 || gpu_id >= sys_stats.gpu_count) return 0;
    if (samples > 60) samples = 60;

    double sum = 0;
    int count = 0;
    for (int i = 0; i < samples; i++) {
        sum += gpu_performance_history[gpu_id][i];
        if (gpu_performance_history[gpu_id][i] > 0) count++;
    }
    return count > 0 ? sum / count : 0;
}
