#!/usr/bin/env python3
"""
Quick Performance Analysis Script

Usage:
    python Real_Inference/analyze_performance.py ./Real_Inference/async_inference_20251112_160832/performance_timings_20251112_160832.jsonl
"""

import json
import sys
import numpy as np
from pathlib import Path


def analyze_timings(json_path: str):
    """Analyze performance timings from JSON or JSON Lines file"""

    # Load JSON or JSON Lines
    timings = []
    with open(json_path, 'r') as f:
        first_line = f.readline()
        f.seek(0)

        # Check if it's JSON Lines (each line is a JSON object)
        if first_line.strip() and not first_line.strip().startswith('['):
            # JSON Lines format
            for line in f:
                if line.strip():
                    timings.append(json.loads(line))
        else:
            # Regular JSON array
            timings = json.load(f)

    if not timings:
        print("⚠️ No timing data found")
        return

    # Extract metrics
    action_times = [t['action_prediction_time'] for t in timings]
    sensor_times = [t['sensor_encoding_time'] for t in timings if t['sensor_encoding_time'] > 0]
    robot_times = [t['robot_encoding_time'] for t in timings if t['robot_encoding_time'] > 0]
    total_times = [t['total_time'] for t in timings]

    # Calculate stats
    def stats(data, name):
        if not data:
            return
        print(f"\n{name}:")
        print(f"  Count: {len(data)}")
        print(f"  Mean:  {np.mean(data):.2f}ms")
        print(f"  Std:   {np.std(data):.2f}ms")
        print(f"  Min:   {np.min(data):.2f}ms")
        print(f"  P50:   {np.percentile(data, 50):.2f}ms")
        print(f"  P95:   {np.percentile(data, 95):.2f}ms")
        print(f"  P99:   {np.percentile(data, 99):.2f}ms")
        print(f"  Max:   {np.max(data):.2f}ms")

    # Print results
    print(f"{'='*80}")
    print(f"Performance Analysis: {Path(json_path).name}")
    print(f"{'='*80}")
    print(f"Total Actions: {len(timings)}")

    # Calculate duration and FPS
    if len(timings) > 1:
        duration = timings[-1]['timestamp'] - timings[0]['timestamp']
        fps = len(timings) / duration if duration > 0 else 0
        print(f"Duration: {duration:.1f}s")
        print(f"Average FPS: {fps:.2f} Hz")

    stats(action_times, "Action Prediction Time")
    stats(sensor_times, "Sensor Encoding Time")
    stats(robot_times, "Robot State Encoding Time")
    stats(total_times, "Total Inference Time")

    # Check for outliers
    p95_action = np.percentile(action_times, 95)
    p99_action = np.percentile(action_times, 99)
    outliers = [t for t in action_times if t > p95_action]

    print(f"\n{'='*80}")
    print(f"Outlier Analysis (>P95: {p95_action:.1f}ms):")
    print(f"  Count: {len(outliers)} / {len(action_times)} ({len(outliers)/len(action_times)*100:.1f}%)")
    if outliers:
        print(f"  Values: {sorted(outliers)[:10]}")  # Show first 10

    # Performance verdict
    print(f"\n{'='*80}")
    print("Performance Verdict:")
    print(f"{'='*80}")

    mean_action = np.mean(action_times)
    p95_total = np.percentile(total_times, 95)

    if p95_total < 30:
        verdict = "✅ EXCELLENT"
        comment = "Meeting 10Hz target comfortably"
    elif p95_total < 50:
        verdict = "✅ GOOD"
        comment = "Can maintain ~10Hz with occasional drops"
    elif p95_total < 100:
        verdict = "⚠️ ACCEPTABLE"
        comment = "May drop below 10Hz during peaks"
    else:
        verdict = "❌ NEEDS OPTIMIZATION"
        comment = "Cannot maintain 10Hz target"

    print(f"{verdict}")
    print(f"Comment: {comment}")
    print(f"Mean Action Time: {mean_action:.1f}ms")
    print(f"P95 Total Time: {p95_total:.1f}ms")
    print(f"Target: 100ms per inference (10Hz)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <performance_timings.json>")
        print("\nExample:")
        print("  python analyze_performance.py async_inference_20251104_234931/performance_timings_20251104_234931.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"❌ Error: File not found: {json_path}")
        sys.exit(1)

    analyze_timings(json_path)
