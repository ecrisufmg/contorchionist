# ctgui.slider

A versatile slider with support for linear, logarithmic, and dB mapping, advanced easing functions, and flexible I/O modes.

## Usage

```pd
[ctgui.slider @min 0 @max 1 @mode lin]
```

## Arguments

- `@min <val>`: Minimum value (default: 0 for lin/log, -120 for db).
- `@max <val>`: Maximum value (default: 1 for lin/log, 12 for db).
- `@mode <mode>`: Mapping mode.
  - `lin`: Linear (default).
  - `log`: Logarithmic.
  - `db`: Decibel (audio fader).
- `@width <val>`: Width in pixels (default: 20).
- `@height <val>`: Height in pixels (default: 120).
- `@color <r g b>`: Handle color (RGB 0-255 or HSB 0-1).
- `@bgcolor <r g b>`: Background color.
- `@slotcolor <r g b>`: Slot/Track color.
- `@markcolor <r g b>`: 0dB mark color (for db mode).
- `@dark`: Enable dark mode theme.
- `@pos`: Enable position mode (2 inlets/outlets).
- `@route`: Enable route mode (single inlet/outlet with tagged messages).
- `@guifps <val>`: GUI refresh rate limit (default: 20).
- `@datafps <val>`: Data output rate limit (default: 0 = unlimited).
- `@linems <val>`: Default line time in ms (default: 0).
- `@linegrain <val>`: Line grain in ms (default: 20).
- `@reasing <mode>`: Default easing function (default: "line").
- `@ctin`: Set input range to 0-127 (MIDI).
- `@bendin`: Set input range to 0-16383 (Pitchbend).
- `@valin <min max>`: Set custom input range.
- `@valout <min max>`: Set custom output range for position outlets.

## I/O Modes

### Default Mode
- **Inlet 1**: Set value (float) or list `target [time] [easing]`.
- **Outlet 1**: Value (float).

### Position Mode (`@pos`)
- **Inlet 1**: Set value (float) or list `target [time] [easing]`.
- **Inlet 2**: Set position (0-1 or custom range).
- **Outlet 1**: Value (float).
- **Outlet 2**: Position (scaled).

### Route Mode (`@route`)
- **Inlet 1**: Accepts tagged messages:
  - `val <target> [time] [easing]`: Set value.
  - `pos <target> [time] [easing]`: Set position.
- **Outlet 1**: Outputs tagged messages:
  - `val <val>`
  - `pos <val>`

## Messages

- `set <val>`: Set value without output.
- `pos <val>`: Set position (visual 0-1).
- `guifps <val>`: Set GUI refresh rate.
- `datafps <val>`: Set data output rate.
- `linems <val>`: Set default line time.
- `reasing <mode>`: Set default easing function.

## Easing Functions

The object supports a wide range of Penner easing functions for smooth transitions. You can specify the easing function by name when sending a list or setting the default `@reasing`.

**Supported Easings:**
- `line` (linear)
- `sine-in`, `sine-out`, `sine-inout`
- `quad-in`, `quad-out`, `quad-inout`
- `cubic-in`, `cubic-out`, `cubic-inout`
- `quart-in`, `quart-out`, `quart-inout`
- `quint-in`, `quint-out`, `quint-inout`
- `sextic-in`, `sextic-out`, `sextic-inout`
- `expo-in`, `expo-out`, `expo-inout`
- `circ-in`, `circ-out`, `circ-inout`
- `back-in`, `back-out`, `back-inout` (supports overshoot param)
- `elastic-in`, `elastic-out`, `elastic-inout` (supports amplitude/period params)
- `bounce-in`, `bounce-out`, `bounce-inout` (supports elasticity param)
- `hann`

**Example:**
To fade to 0.5 over 1000ms with elastic-out easing:
`0.5 1000 elastic-out`
