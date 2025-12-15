#pragma once
//
// viral_config.h
// Wild-only secp256k1 kangaroo "virus" definition
//

#include <cstdint>
#include "Ec.h"
#include "defs.h"

// Types of wild kangaroos we care about
enum ViralType : uint8_t {
    V_WILD1 = 1, // P + HalfRange
    V_WILD2 = 2  // P - HalfRange
};

// A "virus" == one kangaroo walker on secp256k1
struct ViralKangaroo {
    EcPoint point;   // current EC point
    EcInt   dist;    // current distance (scalar)
    uint8_t type;    // ViralType
    uint8_t status;  // 0 = inactive, 1 = active
    uint16_t generation;
    uint16_t virulence;
};
