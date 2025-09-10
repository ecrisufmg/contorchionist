# DSP Audio Processing Workflow

This document outlines the audio signal processing flow, from input to output, using circular buffers and a processing model.

---

## 1. Writing to the Input Buffer

At each DSP cycle, a block of the incoming audio signal is written in its respective input circular buffer (the channel 1 incoming audio signal block is written in the input circular buffer 1, the channel 2 incoming audio signal block is written in the input circular buffer 2, and so on).

> **Obs:** If the `async mode` is disabled, the current incoming audio signal block is sent directly to the model processing and returned to output without intermediate circular buffers (frame-to-frame processing).

---

## 2. Model Triggering and Sliding Window

When all the input circular buffers are filled, the processing model is triggered. After the whole input circular buffer is sent to the model, the oldest block of audio signal is discarded from each input circular buffer (logic of sliding window).

---

## 3. Checking for Batching Possibility

The system checks if batching is possible:

- **3.1. Batching Activated:**
  If `buffer_size > model_buffer_size`, and `buffer_size` is a multiple of `model_buffer_size`, the batching processing is activated. Then each input circular buffer is divided into batches (`num_batch = model_buffer_size / buffer_size`), and a tensor with shape `[num_batches, 1, model_buffer_size]` is created for each input circular buffer.

- **3.2. Batching Not Possible:**
  If `buffer_size <= model_buffer_size`, and `buffer_size` is not a multiple of `model_buffer_size`, the batching processing is not possible. Then, a tensor with shape `[1, buffer_size]` is created for each input circular buffer.

---

## 4. Model Processing

The model processes the input tensors.

> If the `async mode` is enabled, it waits until there is no thread running, calls a new thread, and processes the model.

---

## 5. Model Output and Reshaping

The model returns the output tensors and moves them to the CPU.

- **5.1. If Batching Was Used:**
  It checks if the output is a tuple or a list (multiple output channels). If each output tensor shape is equal to `[num_batches, 1, model_buffer_size]`, it reshapes it to `[1, buffer_size]` and writes to its respective output circular buffer. If the output is a single tensor (one output channel), the same steps are applied to it.

- **5.2. If Batching Was Not Used:**
  It checks if the output is a tuple or a list (multiple output channels), squeezes each output tensor to convert it to 1D, and writes to its respective output circular buffer. If the output is a single tensor (one output channel), the same steps are applied to it.

---

## 6. Consuming from the Output Buffer

At each DSP cycle, a block of samples with a size equal to the block size is consumed from each output circular buffer and sent to the object output signal vectors.

---

## 7. Updating the Sliding Window

In the next DSP cycle, a new block of incoming audio signal is written in each input circular buffer, updating the sliding window with the new signal.

> **Obs:** Once the oldest block of signal was discarded, the input circular buffer will just need one block of incoming audio signal to trigger the model processing again.
