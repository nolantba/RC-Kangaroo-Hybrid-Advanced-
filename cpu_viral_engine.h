#pragma once
//
// cpu_viral_engine.h
// Wild-only secp256k1 kangaroo engine for CPU experiments
//
// This is a header-only class so you can drop it into your RC fork
// and include it from a small test harness.
//

#include <vector>
#include <thread>
#include <atomic>
#include <iostream>
#include <chrono>

#include "viral_config.h"
#include "ram_based_hashtable.h"
#include "Ec.h"
#include "defs.h"
#include "utils.h"

// External globals from RC codebase
extern Ec ec;
extern EcPoint g_G;
extern EcPoint Pnt_HalfRange;
extern EcPoint Pnt_NegHalfRange;
extern EcInt Int_TameOffset;
extern volatile bool gSolved;

class CPUViralEngine {
public:
    CPUViralEngine()
        : range_bits_(0),
          dp_bits_(0),
          EcJumps1_(nullptr),
          EcJumps2_(nullptr),
          EcJumps3_(nullptr),
          population_size_(0),
          worker_count_(0)
    {
        global_virus_index.store(0);
        shutdown_requested.store(false);
        total_operations.store(0);
        total_collisions.store(0);
    }

    ~CPUViralEngine() {
        stop();
    }

    // Prepare engine with same parameters as your main solver
    bool Prepare(const EcPoint& PntToSolve,
                 int range_bits,
                 int dp_bits,
                 EcJMP* EcJumps1,
                 EcJMP* EcJumps2,
                 EcJMP* EcJumps3,
                 size_t population_size)
    {
        PntToSolve_ = PntToSolve;
        range_bits_ = range_bits;
        dp_bits_    = dp_bits;

        EcJumps1_   = EcJumps1;
        EcJumps2_   = EcJumps2;
        EcJumps3_   = EcJumps3;

        shutdown_requested.store(false);
        total_operations.store(0);
        total_collisions.store(0);
        global_virus_index.store(0);

        // HalfRange and A/B like RCCpuKang
        HalfRange_.Set(1);
        HalfRange_.ShiftLeft(range_bits_ - 1);

        PntHalfRange_    = Pnt_HalfRange;
        PntNegHalfRange_ = Pnt_NegHalfRange;

        PntA_ = ec.AddPoints(PntToSolve_, PntHalfRange_);
        PntB_ = ec.AddPoints(PntToSolve_, PntNegHalfRange_);

        // Build population
        population_size_ = population_size;
        population_.clear();
        population_.resize(population_size_);

        initialize_population_wild_only();

        return true;
    }

    // Start worker threads and run until stop(), gSolved, or max_seconds
    void start(size_t worker_threads, double max_seconds = 0.0) {
        stop(); // ensure no old threads

        worker_count_ = (worker_threads == 0) ? 1 : worker_threads;
        shutdown_requested.store(false);

        auto start_time = std::chrono::steady_clock::now();

        worker_threads_.clear();
        worker_threads_.reserve(worker_count_);
        for (size_t i = 0; i < worker_count_; ++i) {
            worker_threads_.emplace_back([this, start_time, max_seconds]() {
                worker_loop(start_time, max_seconds);
            });
        }
    }

    void stop() {
        shutdown_requested.store(true);
        for (auto& t : worker_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
        worker_threads_.clear();
    }

    uint64_t get_total_operations() const {
        return total_operations.load();
    }

    uint64_t get_total_collisions() const {
        return total_collisions.load();
    }

    size_t get_dp_table_size() const {
        return ram_hashtable_.size();
    }

private:
    // Core state
    EcPoint PntToSolve_;
    EcPoint PntA_, PntB_;
    EcPoint PntHalfRange_, PntNegHalfRange_;
    EcInt   HalfRange_;

    int range_bits_;
    int dp_bits_;
    EcJMP* EcJumps1_;
    EcJMP* EcJumps2_;
    EcJMP* EcJumps3_;

    size_t population_size_;
    size_t worker_count_;
    std::vector<ViralKangaroo> population_;

    RAMBasedHashTable ram_hashtable_;

    // threading & stats
    std::vector<std::thread> worker_threads_;
    std::atomic<size_t>  global_virus_index{0};
    std::atomic<bool>    shutdown_requested{false};
    std::atomic<uint64_t> total_operations{0};
    std::atomic<uint64_t> total_collisions{0};

private:
    // ----------------------------------------------------------------------------
    // Population init: wild-only (P + HalfRange) / (P - HalfRange)
    // ----------------------------------------------------------------------------
    void initialize_population_wild_only() {
        for (size_t i = 0; i < population_size_; ++i) {
            ViralKangaroo vk;

            // Alternate WILD1/WILD2
            if (i & 1) {
                vk.type  = V_WILD1;
                vk.point = PntA_; // P + HalfRange
            } else {
                vk.type  = V_WILD2;
                vk.point = PntB_; // P - HalfRange
            }

            vk.dist.SetZero();
            vk.status     = 1;
            vk.generation = 0;
            vk.virulence  = 100;

            // Scatter starting position with a few Jumps1 steps
            int num_jumps = 50 + static_cast<int>(i % 100);
            for (int j = 0; j < num_jumps; ++j) {
                uint32_t jmp_idx = static_cast<uint32_t>(vk.point.x.data[0]) & (JMP_CNT - 1);
                EcJMP& J = EcJumps1_[jmp_idx];
                vk.point = ec.AddPoints(vk.point, J.p);
                vk.dist.Add(J.dist);
            #if USE_GR_EQUIVALENCE
                NormalizePoint_GR(&vk.point);
            #endif
            }

            population_[i] = vk;
        }
    }

    // ----------------------------------------------------------------------------
    // Worker loop: pick walkers using global index and step them forever
    // ----------------------------------------------------------------------------
    void worker_loop(std::chrono::steady_clock::time_point start_time,
                     double max_seconds)
    {
        while (!shutdown_requested.load() && !gSolved) {
            // Hard time limit (for tests)
            if (max_seconds > 0.0) {
                auto now = std::chrono::steady_clock::now();
                double elapsed =
                    std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time)
                        .count();
                if (elapsed >= max_seconds) {
                    break;
                }
            }

            size_t idx = global_virus_index.fetch_add(1, std::memory_order_relaxed);
            if (population_size_ == 0) continue;
            idx %= population_size_;

            process_single_virus(idx);
        }
    }

    // ----------------------------------------------------------------------------
    // One EC step + DP check
    // ----------------------------------------------------------------------------
    void perform_viral_jump(ViralKangaroo& virus) {
        if (virus.status != 1) return;

        uint32_t jmp_idx = static_cast<uint32_t>(virus.point.x.data[0]) & (JMP_CNT - 1);
        EcJMP& J = EcJumps1_[jmp_idx];

        virus.point = ec.AddPoints(virus.point, J.p);
        virus.dist.Add(J.dist);

    #if USE_GR_EQUIVALENCE
        NormalizePoint_GR(&virus.point);
    #endif
    }

    bool is_distinguished_point(const ViralKangaroo& virus) const {
        // Match GPU-style: top bits of x[3] zero
        uint64_t dp_mask64 = ~((1ull << (64 - dp_bits_)) - 1);
        return (virus.point.x.data[3] & dp_mask64) == 0;
    }

    inline void pack_dp_record(const ViralKangaroo& virus, DPRecord& out) const {
        uint8_t* p = out.data;

        // 12-byte X (same layout as RCCpuKang DP buffer)
        std::memcpy(p + 0,  &virus.point.x.data[0], 12);
        std::memset(p + 12, 0, 4);

        // 22-byte dist
        std::memcpy(p + 16, &virus.dist.data[0], 22);

        // type at byte 40
        p[38] = 0;
        p[39] = 0;
        p[40] = virus.type; // V_WILD1 / V_WILD2
        p[41] = 0;

        // padding
        std::memset(p + 42, 0, 6);
    }

    void reseed_virus_as_wild(ViralKangaroo& virus) {
        // Keep its type, restart from A/B and scatter
        const EcPoint& base = (virus.type == V_WILD1) ? PntA_ : PntB_;
        virus.point = base;
        virus.dist.SetZero();
        virus.generation++;

        int num_jumps = 50 + (virus.generation % 100);
        for (int j = 0; j < num_jumps; ++j) {
            uint32_t jmp_idx = static_cast<uint32_t>(virus.point.x.data[0]) & (JMP_CNT - 1);
            EcJMP& J = EcJumps1_[jmp_idx];
            virus.point = ec.AddPoints(virus.point, J.p);
            virus.dist.Add(J.dist);
        #if USE_GR_EQUIVALENCE
            NormalizePoint_GR(&virus.point);
        #endif
        }
    }

    void process_single_virus(size_t index) {
        ViralKangaroo& virus = population_[index];
        if (virus.status != 1) return;

        // Do a small batch to amortize overhead
        const int BATCH_SIZE = 32;
        for (int i = 0; i < BATCH_SIZE && !shutdown_requested.load() && !gSolved; ++i) {
            perform_viral_jump(virus);
            total_operations.fetch_add(1, std::memory_order_relaxed);

            if (is_distinguished_point(virus)) {
                DPRecord rec;
                pack_dp_record(virus, rec);

                // If checkOrAdd returns true, we saw same X with different type
                if (ram_hashtable_.checkOrAdd(rec)) {
                    total_collisions.fetch_add(1, std::memory_order_relaxed);
                    // For now, just log; wiring full solution recovery is phase 2.
                    std::cout << "\n[VIRAL] Collision detected at DP (type="
                              << static_cast<int>(virus.type) << ")\n";
                    // You could set gSolved = true here once you add recovery code.
                }

                // Reseed this walker
                reseed_virus_as_wild(virus);
            }
        }
    }
};
