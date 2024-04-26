import json
import matplotlib.pyplot as plt

import csv
import json

tsv_file = "data/model_emb/run1/history.tsv"
json_file = "data/model_emb/run1/accuracy.json"
save_plot_file = "data/model_emb/run1/plot.png"


def change_file_format():
    with open(tsv_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        data = []
        for row in reader:
            epoch, training_loss, validation_loss, accuracy = row
            data.append(
                {
                    "EPOCH": int(epoch),
                    "training loss": float(training_loss),
                    "validation loss": float(validation_loss),
                    "accuracy": float(accuracy),
                }
            )

    # Write the data to a JSON file
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)


def get_plot():
    data = json.load(open(json_file, "r"))

    epochs = [d["EPOCH"] for d in data]
    training_losses = [d["training loss"] for d in data]
    validation_losses = [d["validation loss"] for d in data]
    accuracies = [d["accuracy"] for d in data]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()

    # Save the figure before showing it
    plt.savefig(save_plot_file, dpi=300)


change_file_format()
get_plot()
