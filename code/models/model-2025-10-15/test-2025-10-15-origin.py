import os
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from T5WithTreeModel import T6ForConditionalGeneration
from gson_reader import CallTreeDataset, read_gsons
import tqdm


batch_size = 16
num_epochs = 20
learning_rate = 2e-5
warmup_steps = 500
output_dir = "../saved_model"  # model output directory

seq_length = 128
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

model_path = "../CodeT5p-220m"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = T6ForConditionalGeneration.from_pretrained(model_path).to(device)
model.init_params()

train_data, test_data = read_gsons(["data_filter_train.json"]), read_gsons(["data_filter_test.json"])

train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)
test_dataset = CallTreeDataset(test_data, tokenizer=tokenizer, max_length=seq_length)



def collate_graphs(x, tokenizer, max_length):

    node_to_sample_ptr = [0]
    edge_to_sample_ptr = [0]
    current = 0
    current_edge = 0
    batch_size = len(x)

    flat_tokens = []
    flat_gat_edges = []
    labels = []
    seq_idx2graph_idxs = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]
        masked_root_method, label = mask_method_method(method_full_name, method_body)
        nodes, edges, num_son = parse(root, 0, [[masked_root_method, 0]], [], 0, 3, 0)

        seq_idx2graph_idx = [item[1] for item in nodes]
        seq_idx2graph_idxs.extend(seq_idx2graph_idx)

        nodes = [item[0] for item in nodes]
        current += len(nodes)
        node_to_sample_ptr.append(current)
        flat_tokens.extend(nodes)
        current_edge += len(edges)
        edge_to_sample_ptr.append(current_edge)

        gat_edge = torch.zeros(2, len(edges), dtype=torch.long)

        for i in range(len(edges)):
            gat_edge[0][i] = -1
            gat_edge[1][i] = -1
            if i < len(edges):
                gat_edge[0][i] = edges[i][0]
                gat_edge[1][i] = edges[i][1]

        flat_gat_edges.append(gat_edge)

        # Convert labels to tensors
        label_tensor = tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        ).input_ids
        labels.append(label_tensor)

    output = tokenizer(
        flat_tokens,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    sample_tensors, attention_masks = output["input_ids"], output["attention_mask"]
    node_to_sample_ptr = torch.tensor(node_to_sample_ptr, dtype=torch.long)
    edge_to_sample_ptr = torch.tensor(edge_to_sample_ptr, dtype=torch.long)
    seq_idx2graph_idxs = torch.tensor(seq_idx2graph_idxs, dtype=torch.long)
    labels = torch.stack(labels, dim=0).squeeze(1)
    gat_edges = flat_gat_edges

    return {"graph": sample_tensors, "label": labels,
            "attention_mask": attention_masks, "gat_edges": gat_edges,
            "node_to_sample_ptr": node_to_sample_ptr, "edge_to_sample_ptr": edge_to_sample_ptr,
            "seq_idx2graph_idx": seq_idx2graph_idxs}


def parse(root, current_idx, nodes, edges, depth, threshold, num_son):
    if "children" not in root.keys() or depth == threshold:
        return nodes, edges, num_son
    children = root["children"]
    if len(children) > 0:
        for i in range(len(children)):
            method = children[i]["method_body"]
            if method == "unResolve":
                continue

            num_son += 1
            nodes.append([method.strip(), num_son])

            edges.append([current_idx, num_son])

            _, _, num_son = parse(children[i], num_son, nodes, edges, depth + 1, threshold, num_son)

    return nodes, edges, num_son


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
        gat_edges = [edge.to(device) for edge in batch["gat_edges"]]
        node_to_sample_ptr = batch["node_to_sample_ptr"].to(device)
        edge_to_sample_ptr = batch["edge_to_sample_ptr"].to(device)
        seq_idx2graph_idx = batch["seq_idx2graph_idx"].to(device)

        # reset gradients
        model.zero_grad()

        # forward pass (compute loss)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # pass labels to compute the loss
            gat_edges=gat_edges,
            node_to_sample_ptr=node_to_sample_ptr,
            edge_to_sample_ptr=edge_to_sample_ptr,
            seq_idx2graph_idx=seq_idx2graph_idx
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
    # if epoch % 2 == 0 or epoch == num_epochs - 1:
    if True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)

print("Training finished!")
