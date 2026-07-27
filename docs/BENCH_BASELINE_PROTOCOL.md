# Benchmark Baseline Capture Protocol

To ensure that performance baselines are reproducible measurements rather than "vibes" captured at a random moment, all baseline refreshes must adhere to the following metrology protocol.

## 1. Machine State
- **Power Profile**: The machine must be connected to AC power. On macOS, "High Power Mode" must be enabled if available for the device.
- **Thermal Soak**: To avoid "turbo boost" artifacts where early runs appear faster than steady-state, perform a **60-second thermal soak**. Run a representative subset of the benchmark suite in a loop for 60 seconds before starting the actual capture.
- **Background Load**: Close all non-essential applications (browsers, IDEs, chat clients) to minimize OS scheduler noise.

## 2. Measurement Process
- **Run Count**: Every kernel must be executed **K ≥ 5 times**.
- **Aggregation**: The reported value must be the **median** of these runs to filter out outliers (OS spikes).
- **Variance**: The **Interquartile Range (IQR)** or standard deviation must be calculated and stored.

## 3. Data Format
Baselines must be stored in JSON format. The root object must include a metadata header to ensure the machine and time are first-class fields.

### Header Requirements
```json
{
  "metadata": {
    "timestamp": "ISO-8601 format",
    "machine_id": "e.g., MacBookPro10,1-M4Max-128GB",
    "os_version": "e.g., macOS 15.1",
    "driver_version": "Metal version",
    "protocol_version": "1.0"
  },
  "results": {
    "kernel_name": {
      "median_ns": 1234.56,
      "variance_ns": 12.3,
      "unit": "ns",
      "iterations": 5
    }
  }
}
```

## 4. Verification
All baseline updates must be accompanied by:
1. The updated JSON file.
2. A link to the capture script used (see `scripts/capture_baseline.sh`).
3. A brief changelog explaining why the baseline is being refreshed (e.g., "New M4 Max hardware" or "Major compiler optimization").