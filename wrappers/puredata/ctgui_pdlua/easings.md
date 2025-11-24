# ctgui.slider Easing Functions

The `ctgui.slider` object supports a wide range of easing functions for smooth transitions (ramps) between values. These can be used to create natural motion, audio fades, or control signal modulation.

## Usage

### 1. Instantiation
You can set the default easing curve when creating the object using the `@reasing` flag (or its aliases `@curve`, `@easing`).

```pd
[ctgui.slider @reasing sine-in]
[ctgui.slider @reasing -3]  ; Ease-Out Cubic
```

### 2. Setting Default Easing
You can change the default easing curve dynamically by sending a `reasing` message. This affects all subsequent ramps that do not specify an explicit easing type.

```pd
[reasing bounce-out(
|
[ctgui.slider]
```

### 3. Triggering Ramps
To trigger a ramp, send a list to the first inlet (for direct value) or second inlet (for normalized 0-1 control).
Format: `[target_value time_ms (easing_type) (param1) (param2) ...]`

- **Standard Ramp:** Uses default easing.
  ```pd
  [0.5 1000(  ; Go to 0.5 over 1000ms
  ```

- **Specific Easing:** Overrides default easing for this ramp.
  ```pd
  [0.5 1000 elastic-out(
  ```

- **Easing with Parameters:** Some curves accept extra parameters.
  ```pd
  [0.5 1000 back-out 3(      ; Back-out with overshoot of 3
  [0.5 1000 elastic-out 1 0.5( ; Elastic-out with Amp=1, Period=0.5
  [0.5 1000 bounce-out 0.8(    ; Bounce-out with high elasticity (0.8)
  ```

---

## Supported Easing Types

### Linear
- `linear`, `line`, `0`: Constant speed.

### Numeric (Power Functions)
- **Positive Numbers (e.g., `3`, `2.5`)**: Ease-In (accelerates). Formula: $t^n$
- **Negative Numbers (e.g., `-3`, `-2.5`)**: Ease-Out (decelerates). Formula: $1 - (1-t)^{|n|}$

### Standard Penner Equations
These functions offer standard animation curves. Available variants: `-in`, `-out`, `-inout`.

- **Sine**: `sine-in`, `sine-out`, `sine-inout`
- **Quad** (Power of 2): `quad-in`, `quad-out`, `quad-inout`
- **Cubic** (Power of 3): `cubic-in`, `cubic-out`, `cubic-inout`
- **Quart** (Power of 4): `quart-in`, `quart-out`, `quart-inout`
- **Quint** (Power of 5): `quint-in`, `quint-out`, `quint-inout`
- **Sextic** (Power of 6): `sextic-in`, `sextic-out`, `sextic-inout`
- **Expo** (Exponential): `expo-in`, `expo-out`, `expo-inout`
- **Circ** (Circular): `circ-in`, `circ-out`, `circ-inout`

### Special Effects

#### Back
Overshoots the target value before settling.
- Types: `back-in`, `back-out`, `back-inout`
- **Parameters:**
  1. `overshoot` (default: 1.70158). Higher values increase the overshoot distance.

#### Elastic
Simulates an elastic band.
- Types: `elastic-in`, `elastic-out`, `elastic-inout`
- **Parameters:**
  1. `amplitude` (default: 1). Magnitude of the oscillation.
  2. `period` (default: 0.3). Duration of one oscillation cycle.

#### Bounce
Simulates a bouncing ball.
- Types: `bounce-in`, `bounce-out`, `bounce-inout`
- **Parameters:**
  1. `elasticity` (default: 0.5). Controls energy preservation (0.0 to <1.0).
     - `0.5`: Standard bounce.
     - `0.8`: Super bouncy (rubber).
     - `0.2`: Heavy (lead).

#### Windowing
- **Hann**: `hann`. Smooth S-curve based on the Hanning window (Cosine).

