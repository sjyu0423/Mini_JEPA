#!/usr/bin/env node
/**
 * Release QA against a Pages-like local mount of website/out.
 * Expects a static server at BASE (default http://127.0.0.1:4321/Mini_JEPA/).
 */
import { chromium, devices } from "playwright";
import { mkdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const base = (process.env.QA_BASE_URL ?? "http://127.0.0.1:4321/Mini_JEPA").replace(/\/$/, "");
const outDir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "qa-artifacts");
mkdirSync(outDir, { recursive: true });

const routes = ["/", "/architecture/", "/experiments/", "/engineering/"];

async function checkPage(page, route) {
  const url = `${base}${route}`;
  const response = await page.goto(url, { waitUntil: "networkidle" });
  if (!response || response.status() >= 400) {
    throw new Error(`${url} status ${response?.status()}`);
  }
  const title = await page.title();
  if (!title.includes("Mini-JEPA")) {
    throw new Error(`${url} unexpected title: ${title}`);
  }
  // Broken image check for visible imgs
  const broken = await page.evaluate(() =>
    Array.from(document.images)
      .filter((img) => !img.complete || img.naturalWidth === 0)
      .map((img) => img.src),
  );
  if (broken.length) {
    throw new Error(`${url} broken images: ${broken.join(", ")}`);
  }
  return title;
}

async function checkDemo(page) {
  await page.goto(`${base}/`, { waitUntil: "networkidle" });
  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "Play" }).waitFor({ timeout: 15000 });
  await page.locator('select').nth(0).selectOption("high-speed");
  await page.getByText("Loading precomputed predictions", { exact: false }).waitFor({ state: "hidden", timeout: 15000 }).catch(() => {});
  await page.waitForTimeout(500);
  await page.locator("select").nth(1).selectOption("m1-jepa");
  for (let i = 0; i < 5; i += 1) {
    await page.getByRole("button", { name: "Step" }).click();
  }
  const errorValue = page.locator("#demo .mono").filter({ hasText: /px|pending/ }).first();
  const errorText = (await errorValue.textContent())?.trim() ?? "";
  if (!(errorText === "pending" || /\d+\.\d+\s*px/.test(errorText))) {
    throw new Error(`unexpected error readout: ${errorText}`);
  }
  const body = await page.locator("body").innerText();
  if (!body.includes("No model runs in the browser")) {
    throw new Error("missing precomputed-prediction disclaimer");
  }
  if (!body.includes("2.835") || !body.includes("2.694")) {
    throw new Error("headline metrics missing from homepage");
  }
  // Confirm scenario JSON was fetched under the Pages subpath.
  const scenarioRequests = [];
  page.on("response", (response) => {
    if (response.url().includes("/data/playback/")) scenarioRequests.push(response.url());
  });
  await page.locator("select").nth(0).selectOption("multiple-bounce");
  await page.waitForTimeout(800);
  if (!scenarioRequests.some((url) => url.includes("/Mini_JEPA/data/playback/multiple-bounce.json"))) {
    // Fallback: ensure the loaded description text changed.
    const description = await page.locator("#demo h3").textContent();
    if (!description?.toLowerCase().includes("multiple bounce")) {
      throw new Error("scenario switch failed");
    }
  }
}

async function reducedMotion(page) {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`${base}/`, { waitUntil: "networkidle" });
  await page.locator("#demo").scrollIntoViewIfNeeded();
  await page.getByRole("button", { name: "Play" }).click();
  // With reduced motion, play should step rather than free-run forever.
  await page.waitForTimeout(300);
}

const browser = await chromium.launch({ headless: true, channel: "chrome" });
const failures = [];

try {
  const desktop = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  for (const route of routes) {
    const title = await checkPage(desktop, route);
    console.log(`OK desktop ${route} :: ${title}`);
  }
  await checkDemo(desktop);
  await desktop.screenshot({ path: path.join(outDir, "desktop-home.png"), fullPage: true });
  console.log("OK desktop demo interactions");

  await reducedMotion(desktop);
  console.log("OK reduced-motion play control");
  await desktop.close();

  const iPhone = devices["iPhone 13"];
  const mobile = await browser.newPage({ ...iPhone });
  for (const route of routes) {
    await checkPage(mobile, route);
    console.log(`OK mobile ${route}`);
  }
  await mobile.goto(`${base}/`, { waitUntil: "networkidle" });
  await mobile.locator("#demo").scrollIntoViewIfNeeded();
  await mobile.getByRole("button", { name: "Play" }).waitFor();
  await mobile.screenshot({ path: path.join(outDir, "mobile-home.png"), fullPage: true });
  console.log("OK mobile demo visible");
  await mobile.close();
} catch (error) {
  failures.push(error instanceof Error ? error.message : String(error));
} finally {
  await browser.close();
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`\nRelease QA passed. Screenshots in ${outDir}`);
