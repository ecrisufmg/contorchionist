

# System Design Specification: Abstract N-Channel Spatial Analyzer

## 1. Project Overview
**Goal:** Implement a new processor `MicArraySpecAnalyzer`, a high-performance C++ DSP library backed by **LibTorch**, wrapped for **Pure Data**.
**Purpose:** Real-time 360° sound source localization using an arbitrary number of microphones ($N$) (optionally calculating their positions regarding Loudspeakers ($M$) around to minimize feedback).
**Input Domain:** Frequency Domain (Magnitude & Phase spectra) provided by external RFFT objects.
**Hardware Context:** Optimized for **Cardioid Microphones** (e.g., Røde NT5), utilizing "Null-Point Steering" (pointing mic tails at loudspeakers) to maximize rejection. Optionally calibratable for other mic patterns (by informing the mics relative own blind spots in angle regarding its capsule direction).

---

## 2. Core Architecture: `MicArraySpectralAnalyzer`

As other core processors in /core/, this class encapsulates all DSP logic and state management, independent of the audio host. The processors use LibTorch tensors and device-agnostic operations to leverage SIMD and GPU acceleration when available.

### A. State Management
The system defines the microphone array as a "Vector Cloud" relative to a centroid $(0,0)$.

**Key Attributes:**
1.  `num_mics` (int): Defined at instantiation.
2.  `fft_size` (int): Size of the incoming spectral frames.
3.  **The Projection Matrix** (`torch::Tensor` shape `[N, 2]`):
    *   A pre-calculated matrix used to instantly convert RMS values into Cartesian ($X, Y$) coordinates.
    *   **Column 0:** $Weight \cdot \sin(\theta_{pickup})$
    *   **Column 1:** $Weight \cdot \cos(\theta_{pickup})$
4.  **Speaker Registry** (`std::vector<float>`): A list of physical angles of the loudspeakers.

### B. Geometry & Calibration
The system distinguishes between **Physical Geometry** and **Analysis Geometry**.

1.  **Input:** User provides `Mic_Index`, `Physical_Angle` (where the capsule points), and `Distance` (meters from center).
2.  **Distance Normalization:**
    *   Find $R_{max}$ (furthest mic).
    *   Calculate Weight $W_i = (r_i / R_{max})$. *(Ensures all mics contribute proportionally to the vector sum, regardless of proximity).*
3.  **Null-Point Steering (Auto-Calibration):**
    *   For Cardioid mics, the "Blind Spot" is at $180^\circ$ (the tail).
    *   **Logic:** For each mic, find the angularly nearest Loudspeaker from the `Speaker Registry`.
    *   **Adjustment:** Physically, the user points the tail at the speaker. Mathematically, the software considers the "Pickup Direction" as the capsule direction.
    *   **Formula:** If the user points the tail at a speaker, the `Physical_Angle` (capsule) is naturally opposite.
    *   *Constraint:* The software uses the `Physical_Angle` for the Vector Math ($X/Y$ projection).

---

## 3. Signal Processing Algorithms

### Step 1: Fractional "Brickwall" Masking
To analyze specific frequency bands with sub-bin precision.

**Function:** `Tensor generate_mask(float low_hz, float high_hz, int sample_rate)`
1.  Convert Hz to fractional bin indices: $b = F \cdot (N_{fft} / SR)$.
2.  Create a mask tensor of size `[1, FFT_Size]`.
3.  For each bin $k$:
    *   Calculate overlap between bin width $[k-0.5, k+0.5]$ and band range $[b_{low}, b_{high}]$.
    *   $Gain_k = \text{OverlapAmount}$ (0.0 to 1.0).
    *   *Result:* A strictly band-limited mask with anti-aliased edges.

### Step 2: Parabolic Interpolation (Frequency Precision)
To find the exact dominant frequency within a band.

**Logic:**
1.  Apply Mask to incoming Magnitude batch.
2.  Sum magnitudes across all channels to find the global peak bin index $i$.
3.  **Detune Calculation ($\delta$):**
    $$\delta = \frac{M_{i+1}^2 - M_{i-1}^2}{2 \cdot (M_{i-1} + M_{i+1} - 2 M_i)}$$
4.  **Frequency Output:** $F_{precise} = (i + \delta) \cdot \frac{SR}{N_{fft}}$.

### Step 3: Tensor-Based Triangulation
To find Angle and Strength using Matrix Multiplication.

**Logic:**
1.  **Batch RMS:** Calculate RMS of the masked band for each mic (Output: Tensor `[1, N]`).
2.  **Projection (The "Magic" substep):**
    Perform a Matrix Multiplication (Dot Product) between the **RMS Vector** `[1, N]` and the **Projection Matrix** `[N, 2]`.
    $$Result_{[1,2]} = RMS_{[1,N]} \times Proj_{[N,2]}$$
    *   $Result[0]$ is total $X$ pull.
    *   $Result[1]$ is total $Y$ pull.
3.  **Polar Conversion:**
    *   $Angle = \operatorname{atan2}(X, Y)$
    *   $Strength = \sqrt{X^2 + Y^2}$

---

## 4. C++ Class Interface (`core_ap_micarrayspecanalyzer.hpp`)

Define it like other core processors in core/include. Use LibTorch tensors for all vector/matrix operations. Define as a header-only library for easy inclusion.

---

## 5. Pure Data Wrapper Specification

**Object Name:** `torch.mic_array_spectral_analyzer~`
**Creation Argument:** `[torch.mic_array_spectral_analyzer~ @mics <num_mics>]`

Please check wrappers/puredata/utils/include to see how to deal with named arguments/flags (using '@' or '-'). Also check the util lib for device handling logic.

### A. Inlet/Outlet Architecture
*   **Dynamic Inlets:**
    *   The object must spawn `num_mics` signal inlets for **Magnitude**.
    *   (Optionally `num_mics` for Phase, though the Core analysis above relies primarily on RMS/Magnitude).
*   **Outlets:**
    *   first outlet: lists of `[band_index (0-based), angle (deg), strength, freq (Hz)]` for each analysis band.

### B. Message/attribute Handlers (if attribute, using @/- syntax when instantiating the object)
*   `mic <id> <angle> <dist>`: Calls `core->set_mic_geometry`.
*   `speaker <angle>`: Calls `core->register_speaker`.
*   `band <low_hz> <high_hz>`: Sets the analysis range.

as message, also accept this:
*   `calibrate`: Triggers `update_projection_matrix`.

### C. The Perform Routine (`DSP Loop`)
1.  Gather input pointers ($N$ arrays).
2.  Pass pointers to `Core::process_band`.
3.  Write resulting `Angle`, `Strength`, `Freq` to the output signal vectors.
    *(Since FFT happens in blocks, the output signal will be "stepped" (constant for the duration of the FFT hop). Use Pd's internal `line~` logic or simply output the step if acceptable).*

