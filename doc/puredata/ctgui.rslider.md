# ctgui.rslider

A versatile range slider GUI object for Pure Data, supporting linear, logarithmic, and dB scales.

## Features

*   **Dual Handle Control**: Select a range between a minimum and maximum value.
*   **Scaling Modes**: Linear (`lin`), Logarithmic (`log`), and Decibel (`db`) scaling.
*   **Flexible I/O**:
    *   **Standard Mode**: Dedicated inlets/outlets for Low and High values.
    *   **Route Mode**: Single inlet/outlet using message selectors (`lo`, `hi`, etc.).
*   **Internal Ramps**: Built-in line generators with easing support for smooth transitions.
*   **Customizable Appearance**: Colors, dimensions, and orientation.

## Usage

```pd
[ctgui.rslider @min 0 @max 100]
[ctgui.rslider @mode db @min -60 @max 6]
[ctgui.rslider @route]
```

## Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `@min <val>` | Minimum value of the range. | 0 (lin/log), -120 (db) |
| `@max <val>` | Maximum value of the range. | 1 (lin/log), 12 (db) |
| `@mode <type>` | Scaling mode: `lin`, `log`, `db`. | `lin` |
| `@width <val>` | Width in pixels. | 130 |
| `@height <val>` | Height in pixels. | 20 |
| `@route` | Enable single inlet/outlet route mode. | Disabled |
| `@pos` | Enable dedicated position I/O (4 inlets/outlets). | Disabled |
| `@posoutput` | Enable normalized position output (3rd outlet in standard mode). | Disabled |
| `@low <val>` | Initial low value. | Min |
| `@high <val>` | Initial high value. | Min |
| `@color <r g b>` | Handle color (RGB or HSB). | Theme dependent |
| `@bgcolor <r g b>` | Background color. | Theme dependent |
| `@slotcolor <r g b>` | Slot/Track color. | Theme dependent |
| `@guifps <val>` | GUI refresh rate limit. | 20 |
| `@datafps <val>` | Data output rate limit (0 = instant). | 0 |
| `@linems <val>` | Default ramp time in ms. | 0 |

## Modes & I/O

### Standard Mode (Default)

*   **Inlet 1 (Left)**: Control **Low** value.
    *   `float`: Set Low value immediately (or ramp if `@linems` > 0).
    *   `pos <val>`: Set Low position (0-1).
    *   `lo <val>`, `lopos <val>`: Explicit control.
*   **Inlet 2 (Right)**: Control **High** value.
    *   `float`: Set High value.
    *   `pos <val>`: Set High position (0-1).
    *   `hi <val>`, `hipos <val>`: Explicit control.
*   **Outlet 1 (Left)**: **Low** value (float).
*   **Outlet 2 (Right)**: **High** value (float).

**With `@pos` flag:**
*   **Inlet 3**: Control **Low** position (0-1).
*   **Inlet 4**: Control **High** position (0-1).
*   **Outlet 3**: **Low** position (0-1).
*   **Outlet 4**: **High** position (0-1).

**With `@posoutput` flag (no `@pos`):**
*   **Outlet 3**: List of positions `{low_pos, high_pos}`.

### Route Mode (`@route`)

*   **Inlet 1**: Accepts tagged messages.
    *   `lo <val>`, `low <val>`: Set Low value.
    *   `hi <val>`, `high <val>`: Set High value.
    *   `lopos <val>`, `hipos <val>`: Set normalized positions.
*   **Outlet 1**: Outputs tagged messages.
    *   `lo <val>`
    *   `hi <val>`
    *   `lopos <val>`
    *   `hipos <val>`

## Ramps & Easing

Values can be ramped over time using the list syntax:
`<target> [time_ms] [easing_mode] [easing_params]`

*   **Example**: `lo 0.5 1000` (Go to 0.5 over 1 second).
*   **Example**: `hi 1.0 2000 sine-in` (Go to 1.0 over 2 seconds with sine-in easing).

Supported easing modes: `linear`, `sine-in`, `sine-out`, `sine-inout`, `quad-in`, `quad-out`, `quad-inout`.

## Mouse Interaction

*   **Click & Drag**: Move the closest handle.
*   **Middle Click / Drag Between**: Move the entire range (both handles).
*   **Click Outside**: Jump closest handle to position.
