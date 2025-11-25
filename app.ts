import { loadLiteRt, loadAndCompile, Tensor, type TypedArray } from "@litertjs/core";

const statusEl = document.getElementById("status") as HTMLDivElement;
const outputEl = document.getElementById("output") as HTMLDivElement;
const runBtn = document.getElementById("runBtn") as HTMLButtonElement;
const uploadBtn = document.getElementById("uploadBtn") as HTMLButtonElement;
const downloadBtn = document.getElementById("downloadBtn") as HTMLButtonElement;
const copyBtn = document.getElementById("copyBtn") as HTMLButtonElement;
const fileInput = document.getElementById("fileInput") as HTMLInputElement;

// Stage update function (no longer needed for visuals, but keeping for potential future use)
function updateStage(stage: string, state: "idle" | "active" | "complete"): void {
	// Visuals removed, but keeping function signature for compatibility
}

function log(message: string): void {
	const timestamp = new Date().toLocaleTimeString();
	outputEl.textContent += `[${timestamp}] ${message}\n`;
	outputEl.scrollTop = outputEl.scrollHeight;
}

let currentModel: Awaited<ReturnType<typeof loadAndCompile>> | null = null;
let lastInferenceResults: Array<{
	name: string;
	data: number[];
	shape: number[];
	dtype: string;
	totalElements: number;
	stats?: {
		min: number;
		max: number;
		mean: number;
		sum: number;
		std: number;
	};
}> | null = null;

async function loadModelFromFile(file: File): Promise<void> {
	try {
		updateStage("compile", "active");
		statusEl.textContent = `Loading model: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
		log(`Loading model: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);

		const arrayBuffer = await file.arrayBuffer();
		const modelData = new Uint8Array(arrayBuffer);

		const model = await loadAndCompile(modelData, {
			accelerator: "wasm",
		});

		currentModel = model;
		updateStage("compile", "complete");
		log("Model loaded successfully");

		// Get model details
		const inputDetails = model.getInputDetails();
		const outputDetails = model.getOutputDetails();

		log(`Input details: ${JSON.stringify(inputDetails, null, 2)}`);
		log(`Output details: ${JSON.stringify(outputDetails, null, 2)}`);

		statusEl.textContent = "Model ready. Click Run Inference to execute.";
		runBtn.disabled = false;
	} catch (error) {
		updateStage("compile", "idle");
		const errorMsg = error instanceof Error ? error.message : String(error);
		statusEl.textContent = `Error loading model: ${errorMsg}`;
		log(`ERROR: ${errorMsg}`);
		console.error(error);
	}
}

async function init(): Promise<void> {
	try {
		updateStage("load", "active");
		statusEl.textContent = "Loading LiteRT Wasm files...";

		// Load Wasm files from server
		await loadLiteRt("/wasm/");

		updateStage("load", "complete");
		log("LiteRT initialized successfully");
		statusEl.textContent = "Ready. Upload a .tflite model file to begin.";

		// Set up file input
		fileInput.onchange = async (e: Event) => {
			const target = e.target as HTMLInputElement;
			const file = target.files?.[0];
			if (file) {
				await loadModelFromFile(file);
			}
		};

		uploadBtn.onclick = () => fileInput.click();

		runBtn.onclick = async () => {
			if (!currentModel) {
				log("ERROR: No model loaded. Please upload a model first.");
				return;
			}
			await runInference(currentModel);
		};
	} catch (error) {
		updateStage("load", "idle");
		const errorMsg = error instanceof Error ? error.message : String(error);
		statusEl.textContent = `Error: ${errorMsg}`;
		log(`ERROR: ${errorMsg}`);
		console.error(error);
	}
}

async function runInference(model: Awaited<ReturnType<typeof loadAndCompile>>): Promise<void> {
	try {
		// Reset output stages
		updateStage("input", "idle");
		updateStage("inference", "idle");
		updateStage("output", "idle");
		lastInferenceResults = null;
		downloadBtn.style.display = "none";
		copyBtn.style.display = "none";

		runBtn.disabled = true;
		updateStage("input", "active");
		statusEl.textContent = "Preparing input tensor...";

		const inputDetails = model.getInputDetails();
		const firstInput = inputDetails[0];

		if (!firstInput) {
			throw new Error("Model has no input details");
		}

		// Convert Int32Array to regular array for shape
		const inputShape = Array.from(firstInput.shape);
		const inputSize = inputShape.reduce((acc: number, val: number) => acc * val, 1);

		// Create input data with correct dtype
		const dtype = firstInput.dtype;
		let inputData: Int32Array | Float32Array;

		if (dtype === "int32") {
			inputData = new Int32Array(inputSize).fill(0); // Fill with 0 for int32
			log(`Creating input tensor with dtype: int32, shape: [${inputShape.join(", ")}]`);
		} else if (dtype === "float32") {
			inputData = new Float32Array(inputSize).fill(0.5); // Fill with 0.5 for float32
			log(`Creating input tensor with dtype: float32, shape: [${inputShape.join(", ")}]`);
		} else {
			throw new Error(`Unsupported input dtype: ${dtype}`);
		}

		const inputTensor = Tensor.fromTypedArray(inputData, inputShape);
		updateStage("input", "complete");
		updateStage("inference", "active");
		statusEl.textContent = "Running inference...";
		log("Running model inference...");
		const outputs = model.run(inputTensor);
		const outputDetails = model.getOutputDetails();
		updateStage("inference", "complete");
		updateStage("output", "active");

		log(`Got ${outputs.length} output(s)`);

		// Process outputs
		for (let i = 0; i < outputs.length; i++) {
			const outputTensor = outputs[i];
			if (!outputTensor) {
				continue;
			}

			// Move to CPU (wasm) only if not already there
			let cpuTensor: Tensor;
			if (outputTensor.accelerator === "wasm") {
				cpuTensor = outputTensor;
			} else {
				cpuTensor = await outputTensor.moveTo("wasm");
			}

			const outputData = cpuTensor.toTypedArray();
			const outputShape = Array.from(outputTensor.type.layout.dimensions);
			const outputDtype = outputTensor.type.dtype;
			const outputName = outputDetails[i]?.name ?? `output_${i}`;
			const totalElements = outputData.length;

			// Calculate statistics directly from TypedArray (avoids stack overflow)
			const stats = getOutputStats(outputData);

			// For large outputs, only store sample data to avoid memory issues
			const MAX_DATA_SAMPLE = 10000;
			const dataSample = totalElements > MAX_DATA_SAMPLE ? Array.from(outputData.slice(0, MAX_DATA_SAMPLE)) : Array.from(outputData);

			// Store results
			if (!lastInferenceResults) {
				lastInferenceResults = [];
			}
			lastInferenceResults.push({
				name: outputName,
				data: dataSample,
				shape: outputShape,
				dtype: outputDtype,
				stats,
				totalElements,
			});

			log(`Output ${i} (${outputName}):`);
			log(`  Shape: [${outputShape.join(", ")}]`);
			log(`  Dtype: ${outputDtype}`);
			log(`  Stats: min=${stats.min.toFixed(4)}, max=${stats.max.toFixed(4)}, mean=${stats.mean.toFixed(4)}, std=${stats.std.toFixed(4)}`);
			log(`  First 10 values: [${Array.from(outputData.slice(0, 10)).join(", ")}...]`);
			log(`  Total elements: ${totalElements}${totalElements > MAX_DATA_SAMPLE ? ` (stored sample of ${MAX_DATA_SAMPLE})` : ""}`);

			// Clean up only if we moved the tensor
			if (outputTensor.accelerator !== "wasm") {
				cpuTensor.delete();
			}
		}

		// Clean up input tensor
		inputTensor.delete();
		updateStage("output", "complete");
		statusEl.textContent = "Inference complete. Results available for download.";
		runBtn.disabled = false;

		// Show download buttons
		downloadBtn.style.display = "inline-block";
		copyBtn.style.display = "inline-block";

		// Show JSON explanation
		const jsonExplanation = document.getElementById("jsonExplanation");
		if (jsonExplanation) {
			jsonExplanation.style.display = "block";
		}
	} catch (error) {
		updateStage("input", "idle");
		updateStage("inference", "idle");
		updateStage("output", "idle");
		const errorMsg = error instanceof Error ? error.message : String(error);
		statusEl.textContent = `Error: ${errorMsg}`;
		log(`ERROR: ${errorMsg}`);
		console.error(error);
		runBtn.disabled = false;
	}
}

// Set up download and copy buttons
downloadBtn.onclick = () => downloadResults();
copyBtn.onclick = () => copyResults();

function downloadResults(): void {
	if (!lastInferenceResults) {
		return;
	}

	// Add metadata to explain what the JSON contains
	const jsonData = {
		metadata: {
			description: "LiteRT.js Model Inference Results",
			timestamp: new Date().toISOString(),
			model: "GPT-2 LiteRT Model",
			explanation: "This JSON contains the raw output tensors from your model inference. Each output includes the tensor data (numbers), shape (dimensions), data type, and statistics.",
		},
		outputs: lastInferenceResults,
		what_is_this: {
			outputs: "Array of model outputs - each represents one output tensor from your model",
			data: "The actual numbers/predictions from the model - these are the raw values",
			shape: "Dimensions of the tensor (e.g., [1, 64, 50257] means 1 batch, 64 positions, 50257 possible tokens)",
			dtype: "Data type: 'float32' for decimal numbers, 'int32' for integers",
			stats: "Statistics calculated from the data: min, max, mean, standard deviation, sum",
			totalElements: "Total number of values in this output tensor",
		},
	};

	const json = JSON.stringify(jsonData, null, 2);
	const blob = new Blob([json], { type: "application/json" });
	const url = URL.createObjectURL(blob);
	const a = document.createElement("a");
	a.href = url;
	a.download = `inference_results_${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
	document.body.appendChild(a);
	a.click();
	document.body.removeChild(a);
	URL.revokeObjectURL(url);
	log("Results downloaded as JSON with metadata");
	log("JSON contains: model outputs, tensor data, shapes, statistics, and explanations");
}

async function copyResults(): Promise<void> {
	if (!lastInferenceResults) {
		return;
	}

	const text = lastInferenceResults
		.map((result, i) => {
			return (
				`Output ${i} (${result.name}):\n` +
				`  Shape: [${result.shape.join(", ")}]\n` +
				`  Dtype: ${result.dtype}\n` +
				`  Data: [${result.data.slice(0, 100).join(", ")}${result.data.length > 100 ? "..." : ""}]\n`
			);
		})
		.join("\n");

	try {
		await navigator.clipboard.writeText(text);
		log("Results copied to clipboard!");
	} catch (error) {
		log(`ERROR: Failed to copy to clipboard: ${error}`);
	}
}

function getOutputStats(data: TypedArray): {
	min: number;
	max: number;
	mean: number;
	sum: number;
	std: number;
} {
	// Calculate stats without spreading arrays (works for large arrays)
	let min = Infinity;
	let max = -Infinity;
	let sum = 0;

	for (let i = 0; i < data.length; i++) {
		const val = Number(data[i]);
		if (!isNaN(val)) {
			if (val < min) min = val;
			if (val > max) max = val;
			sum += val;
		}
	}

	const mean = sum / data.length;

	// Calculate variance
	let variance = 0;
	for (let i = 0; i < data.length; i++) {
		const val = Number(data[i]);
		if (!isNaN(val)) {
			const diff = val - mean;
			variance += diff * diff;
		}
	}
	variance /= data.length;
	const std = Math.sqrt(variance);

	return { min, max, mean, sum, std };
}

// Start initialization when page loads
init();
