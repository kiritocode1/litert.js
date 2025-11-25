# LiteRT.js Model Inference

Run TensorFlow Lite (`.tflite`) models in your browser using Google&apos;s LiteRT.js runtime.

## Quick Start

### 1. Install Dependencies

```bash
bun install
```

### 2. Start the Server

```bash
bun run index.ts
```

Server runs at `http://localhost:3000`

### 3. Open in Browser

Open `http://localhost:3000` in your browser.

### 4. Upload a Model

1. Click **"Upload Model (.tflite)"** button
2. Select a `.tflite` model file
3. Wait for model to load (check status message)

### 5. Run Inference

1. Click **"Run Inference"** button
2. View results in the output log

## Finding Models

Download `.tflite` models from:

-   **[HuggingFace](https://huggingface.co/models?library=tflite)** - Search for TFLite models
-   **[Kaggle](https://www.kaggle.com/models?framework=tfLite)** - Browse TFLite models

## Features

-   ✅ Browser-based ML inference (no server-side processing)
-   ✅ WebGPU acceleration support (Chrome/Edge)
-   ✅ CPU acceleration via XNNPack (all browsers)
-   ✅ File upload interface
-   ✅ Real-time inference results

## How It Works

1. **Server** (`index.ts`) - Serves HTML page and Wasm files
2. **Browser** (`app.ts`) - Loads LiteRT.js runtime and runs models
3. **Models** - Upload `.tflite` files directly in the browser

## Requirements

-   Bun runtime (v1.3.2+)
-   Modern browser with WebAssembly support
-   `.tflite` model files

## Notes

-   Models run entirely in the browser (no data sent to server)
-   WebGPU requires Chrome/Edge (Chromium-based browsers)
-   CPU mode works on all modern browsers
