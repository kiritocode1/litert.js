# LiteRT.js Model Inference - Browser-Based ML Runtime

🎥 **Video Tutorial**: [Watch on YouTube](https://youtu.be/DFZGcZXiiLE?si=4Avs3knkFsnxr4Hz)

Run TensorFlow Lite models directly in your browser using Google&apos;s LiteRT.js runtime. No server-side processing required - everything runs locally in your browser with WebAssembly acceleration.

## 🚀 Features

-   **Browser-Based Inference**: Run ML models entirely in your browser - no data sent to servers
-   **WebAssembly Acceleration**: Fast CPU inference via XNNPack-optimized WASM
-   **WebGPU Support**: GPU acceleration for Chromium-based browsers
-   **GPT-2 Compatible**: Optimized for transformer models like GPT-2
-   **Interactive UI**: Visual pipeline explanation with real-time inference results
-   **Export Results**: Download inference results as JSON or copy to clipboard

## 📋 Requirements

-   Bun runtime (v1.3.2+)
-   Modern browser with WebAssembly support
-   `.tflite` model files

## 🛠️ Installation

```bash
# Install dependencies
bun install

# Start the server
bun run index.ts
```

Server runs at `http://localhost:3000`

## 📖 Usage

1. **Start the server**: `bun run index.ts`
2. **Open browser**: Navigate to `http://localhost:3000`
3. **Upload model**: Click &quot;Upload Model&quot; and select a `.tflite` file
4. **Run inference**: Click &quot;Run Inference&quot; to execute the model
5. **View results**: Check the console output for detailed results
6. **Export data**: Download results as JSON or copy to clipboard

## 🎯 How It Works

The application demonstrates the complete ML inference pipeline:

1. **Load LiteRT**: Initialize WebAssembly runtime files
2. **Compile Model**: Load and compile `.tflite` model for execution
3. **Prepare Input**: Create input tensors matching model requirements
4. **Run Inference**: Execute model on CPU/GPU accelerator
5. **Process Outputs**: Extract and analyze output tensors

## 📊 Understanding the Results

### JSON Output Structure

```json
{
  "metadata": {
    "description": "LiteRT.js Model Inference Results",
    "timestamp": "2025-01-XX...",
    "model": "GPT-2 LiteRT Model"
  },
  "outputs": [
    {
      "name": "Identity",
      "data": [numbers...],
      "shape": [1, 64, 50257],
      "dtype": "float32",
      "stats": {
        "min": -15.0627,
        "max": 16.7778,
        "mean": -0.0060,
        "std": 2.6681
      },
      "totalElements": 3216448
    }
  ]
}
```

### Field Explanations

-   **data**: Raw prediction numbers from the model
-   **shape**: Tensor dimensions `[batch, ...dimensions]`
-   **dtype**: Data type (`float32` for decimals, `int32` for integers)
-   **stats**: Statistics calculated from the output data
-   **totalElements**: Total number of values in the tensor

### For GPT-2 Models

-   **Logits** (`[1, 64, 50257]`): Probability scores for each of 50,257 possible next tokens
-   **Hidden States** (`[1, 2, 12, 64, 64]`): Layer-wise representations from transformer layers

## 🔧 Technical Details

-   **Runtime**: LiteRT.js (Google&apos;s WebAI runtime)
-   **Acceleration**: WebAssembly (XNNPack) for CPU, WebGPU for GPU
-   **Model Format**: TensorFlow Lite (`.tflite`)
-   **Framework**: Bun + TypeScript
-   **UI**: Vanilla HTML/CSS/TypeScript

## 📁 Project Structure

```
mlmodel/
├── index.ts          # Bun server (serves HTML + WASM files)
├── app.html          # Main UI with pipeline explanations
├── app.ts            # Browser-side inference logic
├── package.json      # Dependencies
└── README.md         # This file
```

## 🌐 Finding Models

Download `.tflite` models from:

-   **[HuggingFace](https://huggingface.co/models?library=tflite)** - Search for TFLite models
-   **[Kaggle](https://www.kaggle.com/models?framework=tfLite)** - Browse TFLite models

## 🔒 Privacy

-   All inference runs entirely in your browser
-   No data is sent to external servers
-   Models and results stay on your machine

## 📝 License

MIT License - feel free to use and modify.

## 🔗 Links

-   **Website**: https://aryank.space/
-   **Channel**: BLANK SPACE TECH

## 🙏 Credits

Built with [LiteRT.js](https://ai.google.dev/edge/litert/web) by Google.
