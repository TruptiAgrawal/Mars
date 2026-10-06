import re
import matplotlib.pyplot as plt

log_file = "logs/btcv_liver.log"
output_file = "loss_curve.png"

iterations = []
losses = []

with open(log_file, "r") as f:
    for line in f:
        match = re.search(
            r"Epoch:\s*(\d+).*?Iteration:(\d+), results:.*?[\"']loss[\"']:\s*([0-9.eE+-]+)",
            line
        )

        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            loss = float(match.group(3))

            iterations.append(iteration)
            losses.append(loss)

print(f"Found {len(losses)} loss values")

if not losses:
    print("No loss values found. Check the log file path and format.")
    exit()

plt.figure(figsize=(10, 6))

plt.plot(iterations, losses, marker="o", markersize=2, linewidth=1)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Slide-SAM BTCV Liver Training Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig(output_file, dpi=300)

print(f"Plot saved to: {output_file}")