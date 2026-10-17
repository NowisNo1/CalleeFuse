import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from gson_reader import CallTreeDataset, read_gsons
import tqdm


batch_size = 16
num_epochs = 20
learning_rate = 2e-5
warmup_steps = 500
output_dir = "../saved_model_origin"  # model output directory

seq_length = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = "../CodeT5p-220m"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

train_data, test_data = read_gsons(["data_filter_train.json"]), read_gsons(["data_filter_test.json"])

train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)
test_dataset = CallTreeDataset(test_data, tokenizer=tokenizer, max_length=seq_length)


def collate_graphs(x, tokenizer, max_length):
    batch_size = len(x)
    sample_tensors = []
    labels = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        masked_root_method, label = mask_method_method(method_full_name, method_body)

        # Convert labels to tensors
        label_tensor = tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
        labels.append(label_tensor)
        sample_tensors.append(masked_root_method)

    output = tokenizer(
        sample_tensors,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    sample_tensors, attention_masks = output["input_ids"], output["attention_mask"]
    labels = torch.stack(labels, dim=0).squeeze(1)

    return {"graph": sample_tensors, "label": labels,
            "attention_mask": attention_masks}


def mask_method_method(method_full_name, method_body, mask="<mask>"):
    pos = method_full_name.rfind('.')
    name = method_full_name
    if pos != -1:
        name = method_full_name[pos + 1:]

    return method_body.replace(name, mask).strip(), name


dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # batch size
    shuffle=True,  # shuffle the data
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length)
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,  # batch size
    shuffle=False,  # shuffle the data
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length)
)
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(dataloader) * num_epochs

print(f"total_steps: {total_steps}")
time.sleep(1)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# --------------------------
# 4. Training loop
# --------------------------
model.train()  # switch to training mode

for epoch in range(num_epochs):
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:
        # move the batch to the device
        input_ids = batch["graph"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # reset gradients
        model.zero_grad()
        # forward pass (compute loss)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # pass labels to compute the loss
        )
        loss = outputs.loss  # the loss of a seq2seq model is already in outputs

        # backward pass and parameter update
        loss.backward()
        optimizer.step()
        scheduler.step()

        # accumulate the loss
        total_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item()})

    # compute the average loss for the epoch
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
    time.sleep(0.2)

    # save the model for each epoch (optional)
    if True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)

print("Training finished!")
