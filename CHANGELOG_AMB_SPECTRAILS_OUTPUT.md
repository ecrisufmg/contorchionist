# torch.amb.spectrails~ Output Format Change

## Previous Output Format (Unstable ID)
The control outlet previously output a list of 4 elements, using the **Rank** (magnitude sorting index) as the identifier.

Format: `list <rank> <freq> <mag> <state>`

*   **`<rank>`**: Index of the peak sorted by magnitude (0 = loudest, 1 = second loudest, etc.).
    *   *Issue*: This ID was unstable. If a partial's amplitude changed relative to others, its ID would change, causing the voice allocator to lose track of the note or glitch (glissando artifacts).
*   **`<freq>`**: Frequency in Hz (or MIDI note if `-midi` is active).
*   **`<mag>`**: Magnitude in dB (or Velocity if `-velocity` is active).
*   **`<state>`**: `1` (new), `0` (sustained), `-1` (decayed).

## New Output Format (Stable ID)
The control outlet now outputs a list of 5 elements, using the **Bin Index** (FFT bin) as the identifier.

Format: `list <bin_index> <freq> <mag> <state> <rank>`

*   **`<bin_index>`**: The FFT bin index of the detected peak.
    *   *Benefit*: This ID is stable for the lifetime of the partial. Even if the amplitude fluctuates, the bin index remains constant (or very close), allowing the voice allocator to correctly track and release the note.
*   **`<freq>`**: Frequency in Hz (or MIDI note if `-midi` is active).
*   **`<mag>`**: Magnitude in dB (or Velocity if `-velocity` is active).
*   **`<state>`**: `1` (new), `0` (sustained), `-1` (decayed).
*   **`<rank>`**: The magnitude rank (0 = loudest), provided as the 5th element for informational purposes.
