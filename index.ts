import appHtml from "./app.html";
import { join } from "path";

// Serve LiteRT.js in browser (it requires DOM APIs)
Bun.serve({
	port: 3000,
	routes: {
		"/": appHtml,
		"/wasm/*": async (req) => {
			// Serve wasm files from node_modules
			const url = new URL(req.url);
			const pathParts = url.pathname.split("/wasm/");
			const fileName = pathParts[1];

			if (!fileName) {
				return new Response("File name required", { status: 400 });
			}

			const wasmDir = import.meta.dir ?? ".";
			const wasmPath = join(wasmDir, "node_modules/@litertjs/core/wasm", fileName);

			try {
				const file = Bun.file(wasmPath);
				const exists = await file.exists();
				if (!exists) {
					return new Response("File not found", { status: 404 });
				}

				// Set correct content type for wasm files
				const contentType = fileName.endsWith(".wasm") ? "application/wasm" : fileName.endsWith(".js") ? "application/javascript" : "application/octet-stream";

				return new Response(file, {
					headers: {
						"Content-Type": contentType,
						"Cross-Origin-Embedder-Policy": "require-corp",
						"Cross-Origin-Opener-Policy": "same-origin",
					},
				});
			} catch (error) {
				const errorMsg = error instanceof Error ? error.message : String(error);
				return new Response(`Error: ${errorMsg}`, { status: 500 });
			}
		},
	},
	development: {
		hmr: true,
	},
});

console.log("Server running at http://localhost:3000");
console.log("Open your browser to run LiteRT.js inference");
