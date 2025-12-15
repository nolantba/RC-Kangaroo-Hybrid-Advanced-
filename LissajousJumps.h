// LISSAJOUS JUMP GENERATOR v4.0 - AUTO PATTERNS + BACKWARDS COMPATIBLE
#ifndef LISSAJOUS_JUMP_GENERATOR_HPP
#define LISSAJOUS_JUMP_GENERATOR_HPP

#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <algorithm>
#include <random>

class LissajousJumpGenerator {
public:
    // KEEP YOUR ORIGINAL Config STRUCTURE!
    struct Config {
        double freq_x{1.0};
        double freq_y{1.6180339887498948482};
        double freq_z{2.5066282746310005024};
        double phase_x{0.0};
        double phase_y{M_PI / 2.0};
        double phase_z{M_PI / 4.0};

        // NEW: Auto-pattern parameters (default to classic)
        double damping_x{0.0};
        double damping_y{0.0};
        double damping_z{0.0};
        double mod_freq{0.01};
        double mod_depth{0.0};
        double freq_ratio_xy{1.618};
        double freq_ratio_xz{2.507};

        enum PatternType {
            CLASSIC_LISSAJOUS = 0,     // Your original
            DAMPED_HARMONOGRAPH = 1,   // Spiral patterns
            MODULATED_PATTERN = 2,     // Beating patterns
            MULTI_FREQUENCY = 3,       // Harmonic patterns
            CHAOTIC_MIX = 4           // Random-like
        };

        PatternType pattern_type{CLASSIC_LISSAJOUS}; // Default to your original

        struct Range {
            uint64_t min;
            uint64_t max;
        };

        Range mean_range{10000, 5000000};
        Range std_dev_range{1000, 500000};
        Range skew_range{0, 200};
        size_t table_size{1000000};

        bool validate() const {
            return (table_size > 0) && (freq_x != 0.0) && (freq_y != 0.0) && (freq_z != 0.0) &&
                   (mean_range.min < mean_range.max) &&
                   (std_dev_range.min < std_dev_range.max) &&
                   (skew_range.min < skew_range.max);
        }

        std::string to_string() const {
            const char* pattern_names[] = {"Classic", "Damped", "Modulated", "Multi-Freq", "Chaotic"};
            char buf[512];
            snprintf(buf, sizeof(buf),
                "Pattern: %s | Freq: [%.3f, %.3f, %.3f]\n"
                "Mean: [%lu, %lu] StdDev: [%lu, %lu] Size: %zu",
                pattern_names[pattern_type],
                freq_x, freq_y, freq_z,
                mean_range.min, mean_range.max, std_dev_range.min, std_dev_range.max,
                table_size);
            return buf;
        }
    };

    // Same JumpParams for compatibility
    #pragma pack(push, 1)
    struct JumpParams {
        uint64_t mean;
        uint64_t std_dev;
        int64_t skew;

        uint64_t sample() const {
            return mean;
        }
    };
    #pragma pack(pop)

private:
    Config config_;
    std::vector<JumpParams> jump_table_;
    bool initialized_{false};
    std::mt19937 rng_{std::random_device{}()};

    double map_range(double value, double in_min, double in_max, double out_min, double out_max) const {
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
    }

    // AUTO-PATTERN GENERATION
    void generate_table() {
        jump_table_.clear();
        jump_table_.reserve(config_.table_size);

        std::uniform_real_distribution<double> chaos_dist(-0.1, 0.1);

        for (size_t i = 0; i < config_.table_size; ++i) {
            double t = static_cast<double>(i);
            double x, y, z;

            // AUTO-SELECT PATTERN BASED ON Config
            switch (config_.pattern_type) {
                case Config::CLASSIC_LISSAJOUS:
                    // YOUR ORIGINAL - unchanged
                    x = std::sin(config_.freq_x * t + config_.phase_x);
                    y = std::sin(config_.freq_y * t + config_.phase_y);
                    z = std::sin(config_.freq_z * t + config_.phase_z);
                    break;

                case Config::DAMPED_HARMONOGRAPH:
                    x = std::exp(-config_.damping_x * t) * std::sin(config_.freq_x * t);
                    y = std::exp(-config_.damping_y * t) * std::sin(config_.freq_y * t + M_PI/2);
                    z = std::exp(-config_.damping_z * t) * std::sin(config_.freq_z * t + M_PI/4);
                    break;

                case Config::MODULATED_PATTERN:
                    {
                        double carrier_x = std::sin(config_.freq_x * t);
                        double modulator = 1.0 + config_.mod_depth * std::sin(config_.mod_freq * t);
                        x = carrier_x * modulator;
                        y = std::sin(config_.freq_y * t + M_PI/2) * modulator;
                        z = std::sin(config_.freq_z * t + M_PI/4) * modulator;
                    }
                    break;

                case Config::MULTI_FREQUENCY:
                    x = 0.5 * std::sin(config_.freq_x * t) +
                        0.3 * std::sin(config_.freq_x * config_.freq_ratio_xy * t) +
                        0.2 * std::sin(config_.freq_x * config_.freq_ratio_xz * t);
                    y = 0.5 * std::sin(config_.freq_y * t + M_PI/2) +
                        0.3 * std::sin(config_.freq_y * config_.freq_ratio_xy * t + M_PI/2) +
                        0.2 * std::sin(config_.freq_y * config_.freq_ratio_xz * t + M_PI/2);
                    z = 0.5 * std::sin(config_.freq_z * t + M_PI/4) +
                        0.3 * std::sin(config_.freq_z * config_.freq_ratio_xy * t + M_PI/4) +
                        0.2 * std::sin(config_.freq_z * config_.freq_ratio_xz * t + M_PI/4);
                    break;

                case Config::CHAOTIC_MIX:
                    {
                        double chaos = chaos_dist(rng_);
                        x = std::sin(config_.freq_x * t + chaos);
                        y = std::sin(config_.freq_y * t + M_PI/2 + chaos);
                        z = std::sin(config_.freq_z * t + M_PI/4 + chaos);
                    }
                    break;
            }

            // Normalize
            x = std::max(-1.0, std::min(1.0, x));
            y = std::max(-1.0, std::min(1.0, y));
            z = std::max(-1.0, std::min(1.0, z));

            JumpParams params;
            params.mean = static_cast<uint64_t>(map_range(x, -1.0, 1.0,
                config_.mean_range.min, config_.mean_range.max));
            params.std_dev = static_cast<uint64_t>(map_range(y, -1.0, 1.0,
                config_.std_dev_range.min, config_.std_dev_range.max));
            params.skew = static_cast<int64_t>(map_range(z, -1.0, 1.0,
                config_.skew_range.min, config_.skew_range.max));

            jump_table_.push_back(params);
        }
    }

    // AUTO-CONFIGURE PATTERN BASED ON PROBLEM SIZE
    void auto_configure(size_t problem_size) {
        if (problem_size > 1000000000) {
            // Large problem -> Damped for exploration
            config_.pattern_type = Config::DAMPED_HARMONOGRAPH;
            config_.damping_x = 0.0001;
            config_.damping_y = 0.00015;
            config_.damping_z = 0.0002;
        } else if (problem_size < 100000) {
            // Small problem -> Multi-frequency for intensity
            config_.pattern_type = Config::MULTI_FREQUENCY;
            config_.freq_ratio_xy = 2.0;
            config_.freq_ratio_xz = 3.0;
        } else {
            // Medium problem -> Modulated for balance
            config_.pattern_type = Config::MODULATED_PATTERN;
            config_.mod_freq = 0.005;
            config_.mod_depth = 0.3;
        }
    }

public:
    // YOUR ORIGINAL CONSTRUCTORS - NOW WITH AUTO-PATTERNS!
    LissajousJumpGenerator() : LissajousJumpGenerator(1000000) {}

    LissajousJumpGenerator(size_t size) {
        Config config;
        config.table_size = size;

        // AUTO-CONFIGURE PATTERN based on size
        auto_configure(size);

        initialize(config);
    }

    explicit LissajousJumpGenerator(const Config& config) {
        initialize(config);
    }

    bool initialize(const Config& config) {
        if (!config.validate()) {
            std::cerr << "Invalid Lissajous configuration" << std::endl;
            return false;
        }

        config_ = config;
        generate_table();
        initialized_ = true;

        return true;
    }

    // ORIGINAL INTERFACE
    const JumpParams& get_jump_params(uint64_t index) const {
        return jump_table_[index % jump_table_.size()];
    }

    size_t get_table_size() const { return jump_table_.size(); }
    bool is_initialized() const { return initialized_; }
    const Config& config() const { return config_; }
    const JumpParams* data() const { return jump_table_.data(); }
    size_t memory_size() const { return sizeof(JumpParams) * jump_table_.size(); }

    // ORIGINAL PRESETS
    static Config default_config() {
        return Config{};
    }

    static Config optimized_config(size_t problem_bits) {
        Config config = default_config();
        double scale = std::pow(2.0, (static_cast<double>(problem_bits) - 64.0) / 4.0);
        config.mean_range.min = static_cast<uint64_t>(config.mean_range.min * scale);
        config.mean_range.max = static_cast<uint64_t>(config.mean_range.max * scale);
        config.std_dev_range.min = static_cast<uint64_t>(config.std_dev_range.min * scale);
        config.std_dev_range.max = static_cast<uint64_t>(config.std_dev_range.max * scale);
        config.table_size = std::min(static_cast<size_t>(10000000), static_cast<size_t>(config.table_size * scale));

        // AUTO: Also optimize pattern based on problem size
        if (problem_bits > 70) {
            config.pattern_type = Config::DAMPED_HARMONOGRAPH;
            config.damping_x = 0.0001;
        } else if (problem_bits < 50) {
            config.pattern_type = Config::MULTI_FREQUENCY;
            config.freq_ratio_xy = 2.0;
        } else {
            config.pattern_type = Config::MODULATED_PATTERN;
            config.mod_depth = 0.2;
        }

        return config;
    }

    // MANUAL PATTERN SELECTION (OPTIONAL)
    static Config chaotic_config() {
        Config config = default_config();
        config.pattern_type = Config::CHAOTIC_MIX;
        return config;
    }

    static Config harmonograph_config() {
        Config config = default_config();
        config.pattern_type = Config::DAMPED_HARMONOGRAPH;
        config.damping_x = 0.0001;
        config.damping_y = 0.00015;
        config.damping_z = 0.0002;
        return config;
    }
};

#endif // LISSAJOUS_JUMP_GENERATOR_HPP
