/**
 * Renders the checks in `verify.html` on a real GPU and prints what came out.
 *
 * The CI gate is pure TypeScript and needs no GPU, which leaves the integrator itself unchecked: a wrong density or a
 * misread texel compiles perfectly and simply produces the wrong picture. These checks put numbers on that. Every one
 * of them renders a scene whose answer is known on paper, so a failure says which part is wrong rather than that
 * something changed.
 *
 * Headless Chrome will not run WebGPU without the flags below, and `--dump-dom` fires before the device has even
 * answered, so the page posts its result back to this server instead.
 *
 * Usage: pnpm verify [--keep] [--port 8765] [--page verify.html]
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..");

const argument = (name, fallback) => {
  const at = process.argv.indexOf(`--${name}`);

  return at >= 0 && process.argv[at + 1] ? process.argv[at + 1] : fallback;
};

const port = Number(argument("port", 8765));
const keep = process.argv.includes("--keep");
// Another page under scripts/, for chasing one check without running all of them.
const page = argument("page", "verify.html");

/** Where a browser that can run WebGPU might be, newest first. */
const BROWSERS = [
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
  "/Applications/Chromium.app/Contents/MacOS/Chromium",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
  "/usr/bin/chromium-browser",
];

const browser = process.env.CHROME ?? BROWSERS.find(candidate => fs.existsSync(candidate));

if (!browser) {
  console.error("No Chrome or Chromium found. Set CHROME to one that can run WebGPU.");
  process.exit(1);
}

if (!fs.existsSync(path.join(root, "dist", "index.js"))) {
  console.error("Build first: pnpm build");
  process.exit(1);
}

const TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".mjs": "text/javascript",
  ".json": "application/json",
  ".wasm": "application/wasm",
};

let settled = false;

const finish = (code, message) => {
  if (settled) return;

  settled = true;

  if (message) console.log(message);

  server.close();
  child?.kill();
  process.exit(code);
};

const server = http.createServer((request, response) => {
  if (request.method === "POST") {
    const chunks = [];

    request.on("data", chunk => chunks.push(chunk));
    request.on("end", () => {
      const body = Buffer.concat(chunks).toString();

      response.end("ok");

      if (request.url === "/log") {
        if (process.env.VERBOSE) console.log(`  … ${body}`);

        return;
      }

      // A check that did not meet its expectation marks itself, and the exit code follows.
      finish(body.includes("<-- OFF") || body.startsWith("EXC") || body.startsWith("ERROR") ? 1 : 0, body);
    });

    return;
  }

  const url = request.url.split("?")[0];
  const file = url === "/" ? path.join(here, page) : path.join(root, url);

  if (!path.resolve(file).startsWith(root)) {
    response.statusCode = 403;
    response.end();

    return;
  }

  try {
    response.setHeader("content-type", TYPES[path.extname(file)] ?? "application/octet-stream");
    response.end(fs.readFileSync(file));
  } catch {
    response.statusCode = 404;
    response.end();
  }
});

server.listen(port);

const profile = path.join(root, "node_modules", ".cache", "verify-profile");

const child = spawn(browser, [
  ...keep ? [] : ["--headless=new"],
  "--enable-unsafe-webgpu",
  "--enable-features=WebGPU",
  "--no-sandbox",
  `--user-data-dir=${profile}`,
  `http://127.0.0.1:${port}/`,
], { stdio: "ignore" });

child.on("error", error => finish(1, `Could not start the browser: ${error.message}`));

setTimeout(() => finish(1, "The page never reported back. Run with --keep to watch it in a window."), 1800000);
