# ctgui.slider

A versatile GUI slider object for Pure Data with support for linear, logarithmic, and dB scales, advanced easing functions for automation, and customizable appearance.

## Instantiation

```pd
[ctgui.slider @arg1 val1 @flag ...]
```

### Arguments

#### Core Configuration
- `@min <val>`: Minimum value. Default: `0` (lin/log), `-120` (db).
- `@max <val>`: Maximum value. Default: `1` (lin/log), `12` (db).
- `@mode <mode>`: Scale mode.
  - `lin`: Linear (default).
  - `log`: Logarithmic / Exponential.
  - `db`: dB Fader scale (audio friendly).
- `@width <val>` / `@w`: Width in pixels (default: 20).
- `@height <val>` / `@h`: Height in pixels (default: 120).
- `@vert` / `@vertical`: Force vertical orientation.
- `@horiz` / `@horizontal`: Force horizontal orientation.

#### Input/Output Scaling
- `@ctin`: Sets control input/output range to MIDI 0-127 (Inlet 2 / Outlet 2).
- `@bendin`: Sets control input/output range to 14-bit MIDI 0-16383.
- `@valin <min> <max>`: Custom range for Inlet 2 (Control Input). Default: `0 1`.
- `@valout <min> <max>`: Custom range for Outlet 2 (Control Output). Default: `0 1`.

#### Animation & Easing
- `@reasing <type>` / `@curve` / `@easing`: Default easing curve for ramps. Default: `line`.
- `@linems <ms>`: Default time for ramps when receiving a float. Default: `0` (instant).
- `@linegrain <ms>`: Update rate for ramps. Default: `20`.

#### Appearance
- `@color <r g b>` / `@handlecolor`: Handle color (RGB or HSB).
- `@bgcolor <r g b>` / `@bg`: Background color.
- `@slotcolor <r g b>` / `@trackcolor`: Slot/Track color.
- `@markcolor <r g b>` / `@mark`: Color of the 0dB mark (in dB mode).
- `@dark`: Enable dark mode theme.
- `@guifps <val>` / `@fps`: GUI refresh rate limit. Default: `20`.

#### Data Rate
- `@datafps <val>`: Limit data output rate (messages per second). Default: `0` (unlimited).

---

## Inlets & Messages

### Inlet 1: Main Control (Value)
- **Float**: Sets the slider value. If `@linems` > 0, ramps to value. Outputs value.
- **List**: `[target_value time_ms (easing) (param1) (param2) ...]`
  - Triggers a ramp to `target_value` over `time_ms`.
  - Optional `easing` overrides the default curve.
  - Optional `params` configure complex curves (e.g., elasticity).
- **`set <float>`**: Sets value instantly without output.
- **`reasing <type>`**: Changes the default easing curve.
- **`guifps <val>`**: Sets GUI refresh rate.
- **`datafps <val>`**: Sets data output rate limit.

### Inlet 2: Normalized Control (Visual)
- **Float**: Sets value based on normalized position (0-1, or scaled by `@valin`).
  - Useful for connecting to other GUI elements or MIDI.
- **List**: `[norm_target time (easing) ...]`
  - Ramps to a value corresponding to the normalized target.

---

## Outlets

- **Outlet 1**: Current value (float). Scaled between `@min` and `@max`.
- **Outlet 2**: Control value (float). Scaled between `@valout` min/max (default 0-1).

---

## Easing Functions

The object supports a robust animation engine using Robert Penner's easing equations.

### Usage
Specify the easing type in the list message or via `@reasing`.

```pd
[0.5 1000 elastic-out(  -> Go to 0.5 in 1s using elastic-out
```

### Supported Types

#### Standard
- `linear` (or `line`, `0`)
- `sine-in`, `sine-out`, `sine-inout`
- `quad-in`, `quad-out`, `quad-inout`
- `cubic-in`, `cubic-out`, `cubic-inout`
- `quart-in`, `quart-out`, `quart-inout`
- `quint-in`, `quint-out`, `quint-inout`
- `sextic-in`, `sextic-out`, `sextic-inout`
- `expo-in`, `expo-out`, `expo-inout`
- `circ-in`, `circ-out`, `circ-inout`

#### Numeric (Power)
- Positive float (e.g., `2.5`): Ease-In power curve.
- Negative float (e.g., `-2.5`): Ease-Out power curve.

#### Special
- **Back**: `back-in`, `back-out`, `back-inout`
  - *Param 1*: Overshoot amount (default ~1.7).
- **Elastic**: `elastic-in`, `elastic-out`, `elastic-inout`
  - *Param 1*: Amplitude (default 1).
  - *Param 2*: Period (default 0.3).
- **Bounce**: `bounce-in`, `bounce-out`, `bounce-inout`
  - *Param 1*: Elasticity (0.0 - 1.0). Default 0.5.
    - `> 0.5`: More bouncy.
    - `< 0.5`: Less bouncy.
- **Window**: `hann` (Cosine window shape).
