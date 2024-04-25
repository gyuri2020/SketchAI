import argparse
import multiprocessing as mp
import numpy as np
import os
import shutil
import torch
import yaml
from tqdm import tqdm

from PIL import Image
from transformers import CLIPModel, CLIPProcessor, AdamW, get_scheduler
from torch.utils.data import DataLoader, RandomSampler, random_split


os.environ["TOKENIZERS_PARALLELISM"] = "false"


from sketchai_train.train_dataset import (
    ImageCaptionDataset,
    ImageCaptionCollator,
)


def do_train(model, train_dl):
    train_loss = 0
    model.train()
    for bid, (batch, _) in enumerate(train_dl):
        if bid % 100 == 0:
            print("...{:d} training steps complete".format(bid))
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, return_loss=True)
        loss = outputs.loss
        train_loss += loss.detach().cpu().numpy()
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    print("...{:d} training steps COMPLETE".format(bid))
    return train_loss


def do_eval(model, eval_dl):
    model.eval()
    val_loss, val_acc, num_examples = 0, 0, 0
    for bid, (batch, _) in enumerate(eval_dl):
        if bid % 100 == 0:
            print("... {:d} validation steps complete".format(bid))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_loss=True)

        loss = outputs.loss
        val_loss += loss.detach().cpu().numpy()

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predictions = torch.argmax(probs, dim=-1)
        labels = torch.arange(len(predictions)).to(device)

        accuracy = torch.sum(predictions == labels)
        num_examples += len(predictions)
        val_acc += accuracy

    print("... {:d} validation steps COMPLETE".format(bid))
    val_acc = val_acc.detach().cpu().numpy() / num_examples
    return val_loss, val_acc


def save_checkpoint(model, model_dir, epoch):
    model.save_pretrained(os.path.join(model_dir, "ckpt-{:d}".format(epoch + 1)))


def save_training_history(history, model_dir):
    fhist = open(os.path.join(model_dir, "history.tsv"), "w")
    for epoch, train_loss, val_loss, val_acc in history:
        fhist.write(
            "{:d}\t{:.5f}\t{:.5f}\t{:.5f}\n".format(
                epoch, train_loss, val_loss, val_acc
            )
        )
    fhist.close()


###################### main ###########################


open_ai_clip_model_path = "openai/clip-vit-base-patch32"
# https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", help="provides training params")
args = parser.parse_args()

config_file = args.config_file
with open(config_file, "r") as fcfg:
    config = yaml.full_load(fcfg)
print(f"config: {config}\n\n\n\n")


epoch_range = range(0)
model_path = None

model_path = open_ai_clip_model_path
num_epochs = config["num_epochs"]
epoch_range = range(num_epochs)

model_dir = os.path.join(
    config["models_dir"], os.path.basename(config_file).split(".")[0]
)
print("model dir: ", model_dir)
# shutil.rmtree(model_dir, ignore_errors=True)
# os.makedirs(model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# ----------------------------
# load model and processor
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(open_ai_clip_model_path)
# ----------------------------

# ----------------------------
# load dataset


train_ds = ImageCaptionDataset()
collator = ImageCaptionCollator(processor)


# ----------------------------

# Compute the lengths of the splits
total_size = len(train_ds)
train_size = int(0.8 * total_size)
valid_size = total_size - train_size

# Create the splits
train_ds, valid_ds = random_split(train_ds, [train_size, valid_size])

# Create the samplers
train_sampler = RandomSampler(train_ds)
valid_sampler = RandomSampler(valid_ds)


train_dl = DataLoader(
    train_ds,
    batch_size=config["train_batch_size"],
    # shuffle=True,
    sampler=train_sampler,
    num_workers=7,
    collate_fn=collator,
)
validation_dl = DataLoader(
    valid_ds,
    batch_size=config["validation_batch_size"],
    sampler=valid_sampler,
    num_workers=7,
    collate_fn=collator,
)

optimizer = AdamW(model.parameters(), lr=config["learning_rate"])

num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
assert device == torch.device("cuda")
torch.cuda.empty_cache()
model.to(device)

history = []

for epoch in tqdm(epoch_range):
    try:
        train_loss = do_train(model, train_dl)
        val_loss, val_acc = do_eval(model, validation_dl)
        save_checkpoint(model, model_dir, epoch)
        save_model_path = os.path.join(model_dir, f"ckpt-{epoch + 1}")

        history.append((epoch + 1, train_loss, val_loss, val_acc))
        print(
            "EPOCH {:d}, training loss: {:.3f}, validation loss: {:.3f}, accuracy: {:.3f}".format(
                epoch + 1, train_loss, val_loss, val_acc
            )
        )
    except Exception as e:
        print("exception: ", e)
        continue
save_training_history(history, model_dir)
