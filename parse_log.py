from statistics import mean
import re
import csv

log_file_path = "log.txt"
output_csv_path = "parsed_log.csv"

epochs = []
losses = []
forward_times = []
backprop_times = []

epoch_pattern = r"Epoch (\d+), Loss: ([\d.]+)"
forward_time_pattern = r"Total Forward Pass Time: ([\d.]+) seconds"
backprop_time_pattern = r"Total Backpropagation Time: ([\d.]+) seconds"

with open(log_file_path, "r") as file:
    lines = file.readlines()

for i in range(len(lines)):
    match_epoch = re.search(epoch_pattern, lines[i])
    if match_epoch:
        epochs.append(int(match_epoch.group(1)))
        losses.append(float(match_epoch.group(2)))
        
        match_forward = re.search(forward_time_pattern, lines[i + 1])
        forward_times.append(float(match_forward.group(1)) if match_forward else None)
        
        match_backprop = re.search(backprop_time_pattern, lines[i + 2])
        backprop_times.append(float(match_backprop.group(1)) if match_backprop else None)

with open(output_csv_path, mode="w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Epoch", "Loss", "Forward Pass Time", "Backprop Time"])
    for epoch, loss, forward, backprop in zip(epochs, losses, forward_times, backprop_times):
        csv_writer.writerow([epoch, loss, forward, backprop])
print(f"Average Forward {mean(forward_times)}, Average Backward {mean(backprop_times)}")
print(f"Parsed data saved to {output_csv_path}")
