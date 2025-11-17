#include "SC_PlugIn.h"

// Undefine the conflicting macro 'Print' from SuperCollider headers
#ifdef Print
#undef Print
#endif

#include "../../../core/include/core_ap_rmsoverlap.h"
#include "../../../core/include/core_util_windowing.h"

// InterfaceTable contains pointers to functions in the host (server).
static InterfaceTable *ft;

// Use the contorchionist namespace
using RMSOverlapCore = contorchionist::core::ap_rmsoverlap::RMSOverlap<float>;
using WindowType = contorchionist::core::util_windowing::Type;
using WindowAlignment = contorchionist::core::util_windowing::Alignment;
using NormalizationType = RMSOverlapCore::NormalizationType;

// Declare struct to hold unit generator state
struct RMSOverlap : public Unit {
    RMSOverlapCore* m_analyzer;

    // Parameters stored for re-initialization if needed
    int    m_initial_window_size;
    int    m_initial_hop_size;
    WindowType m_initial_win_type;
    float  m_initial_zero_padding_factor;
    WindowAlignment m_initial_win_align;
    NormalizationType m_initial_norm_type;
    float  m_initial_fixed_norm_multiplier;
};

// Forward declarations of UGen functions
static void RMSOverlap_Ctor(RMSOverlap* unit);
static void RMSOverlap_Dtor(RMSOverlap* unit);
static void RMSOverlap_next_a(RMSOverlap* unit, int inNumSamples);

// --- Helper function to map float to WindowType ---
static WindowType float_to_window_type(float f) {
    int val = static_cast<int>(f);
    switch (val) {
        case 0: return WindowType::RECTANGULAR;
        case 1: return WindowType::HANN;
        case 2: return WindowType::BARTLETT; // Triangular
        case 3: return WindowType::HAMMING;
        case 4: return WindowType::BLACKMAN;
        case 5: return WindowType::COSINE;
        default: return WindowType::HANN; // Default
    }
}

// --- Helper function to map float to WindowAlignment ---
static WindowAlignment float_to_window_alignment(float f) {
    int val = static_cast<int>(f);
    switch (val) {
        case 0: return WindowAlignment::LEFT;
        case 1: return WindowAlignment::CENTER;
        case 2: return WindowAlignment::RIGHT;
        default: return WindowAlignment::CENTER; // Default
    }
}

// --- Helper function to map float to NormalizationType ---
static NormalizationType float_to_normalization_type(float f) {
    int val = static_cast<int>(f);
    switch (val) {
        case 0: return NormalizationType::WINDOW_OVERLAP_RMS;
        case 1: return NormalizationType::WINDOW_OVERLAP_MEAN;
        case 2: return NormalizationType::WINDOW_OVERLAP_VALS;
        case 3: return NormalizationType::OVERLAP_INVERSE;
        case 4: return NormalizationType::FIXED_MULTIPLIER;
        case 5: return NormalizationType::NONE;
        default: return NormalizationType::WINDOW_OVERLAP_RMS; // Default
    }
}

// --- UGen Constructor ---
void RMSOverlap_Ctor(RMSOverlap* unit) {
    SETCALC(RMSOverlap_next_a);

    // --- Get arguments passed from SuperCollider ---
    unit->m_initial_window_size = static_cast<int>(IN0(1));
    unit->m_initial_hop_size    = static_cast<int>(IN0(2));
    unit->m_initial_win_type    = float_to_window_type(IN0(3));
    unit->m_initial_zero_padding_factor = IN0(4);
    unit->m_initial_win_align   = float_to_window_alignment(IN0(5));
    unit->m_initial_norm_type   = float_to_normalization_type(IN0(6));
    unit->m_initial_fixed_norm_multiplier = IN0(7);

    // --- Validate parameters ---
    if (unit->m_initial_window_size <= 0) {
        unit->m_initial_window_size = 1024;
        if(ft->fPrint) ft->fPrint("RMSOverlap: Invalid window size, defaulting to %d\n", unit->m_initial_window_size);
    }
    if (unit->m_initial_hop_size <= 0) {
        unit->m_initial_hop_size = unit->m_initial_window_size / 2;
         if(ft->fPrint) ft->fPrint("RMSOverlap: Invalid hop size, defaulting to %d\n", unit->m_initial_hop_size);
    }
    if (unit->m_initial_hop_size > unit->m_initial_window_size) {
        unit->m_initial_hop_size = unit->m_initial_window_size;
        if(ft->fPrint) ft->fPrint("RMSOverlap: Hop size > window size, setting hop to window size (%d)\n", unit->m_initial_hop_size);
    }
    if (unit->m_initial_zero_padding_factor < 0.f || unit->m_initial_zero_padding_factor >= 1.f) {
        unit->m_initial_zero_padding_factor = 0.f;
        if(ft->fPrint) ft->fPrint("RMSOverlap: Invalid zero padding factor, defaulting to 0.0\n");
    }

    // --- Initialize the RMSOverlapCore ---
    int sc_block_size = unit->mRate->mBufLength;

    try {
        unit->m_analyzer = new RMSOverlapCore(
            unit->m_initial_window_size,
            unit->m_initial_hop_size,
            unit->m_initial_win_type,
            unit->m_initial_zero_padding_factor,
            unit->m_initial_win_align,
            unit->m_initial_norm_type,
            unit->m_initial_fixed_norm_multiplier,
            sc_block_size
        );
    } catch (const std::exception& e) {
        if(ft->fPrint) ft->fPrint("RMSOverlap: Failed to create RMSOverlapCore: %s\n", e.what());
        unit->m_analyzer = nullptr;
        SETCALC(ClearUnitOutputs);
        return;
    }

    if (!unit->m_analyzer) {
        if(ft->fPrint) ft->fPrint("RMSOverlap: Failed to create RMSOverlapCore instance!\n");
        SETCALC(ClearUnitOutputs);
        return;
    }

    // Calculate one sample of output for initialization
    RMSOverlap_next_a(unit, 1);
}

// --- UGen Destructor ---
void RMSOverlap_Dtor(RMSOverlap* unit) {
    if (unit->m_analyzer) {
        delete unit->m_analyzer;
        unit->m_analyzer = nullptr;
    }
}

// --- UGen Calculation Function (Audio Rate) ---
void RMSOverlap_next_a(RMSOverlap* unit, int inNumSamples) {
    float* in = IN(0);
    float* out = OUT(0);

    if (!unit->m_analyzer) {
        for (int i = 0; i < inNumSamples; ++i) {
            out[i] = 0.0f;
        }
        return;
    }

    // Post input data to the circular buffer
    unit->m_analyzer->post_input_data(in, inNumSamples);

    // Process data from the circular buffer to get output
    unit->m_analyzer->process(nullptr, out, inNumSamples);
}

// --- Plugin Load Function ---
PluginLoad(RMSOverlapUGens) {
    ft = inTable;
    DefineDtorUnit(RMSOverlap);
}
