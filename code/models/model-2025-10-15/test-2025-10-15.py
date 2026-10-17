import os
import random
import time

import numpy as np
import torch
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, T5ForConditionalGeneration, CodeGenForCausalLM, PLBartForConditionalGeneration
from T5WithTreeModel import T6ForConditionalGeneration
from gson_reader import CallTreeDataset, read_gsons
import tqdm
from torch.amp import GradScaler



batch_size = 16
num_epochs = 15
warmup_steps = 2000
max_nodes = 20
output_dir = "../CodeT5p_no_mask"
seq_length = 128
label_length = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "../CodeT5p-220m"


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def save_checkpoint(epoch, model, optimizer, scheduler, batch_size, trainable_params, warmup_steps, seq_length, max_label_length, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state().cpu(),
        'torch_cuda_random_state': torch.cuda.get_rng_state().cpu(),
        'hyperparams': {
            'batch_size': batch_size,
            'trainable_params': trainable_params,
            'warmup_steps': warmup_steps,
            'seq_length': seq_length,
            'label_length': max_label_length,
            'max_nodes': max_nodes
        }
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path} (epoch: {epoch})")


def load_checkpoint(load_path, model, optimizer, scheduler):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Checkpoint {load_path} not found!")

    checkpoint = torch.load(load_path, map_location='cuda' if torch.cuda.is_available() else 'cpu', weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    random.setstate(checkpoint['python_random_state'])
    np.random.set_state(checkpoint['numpy_random_state'])
    torch_rng_state = checkpoint['torch_random_state']

    if not isinstance(torch_rng_state, torch.ByteTensor):
        torch_rng_state = torch_rng_state.to(dtype=torch.uint8, device='cpu')
    torch.set_rng_state(torch_rng_state)

    if torch.cuda.is_available():
        cuda_rng_state = checkpoint['torch_cuda_random_state']
        if not isinstance(cuda_rng_state, torch.ByteTensor):
            cuda_rng_state = cuda_rng_state.to(dtype=torch.uint8, device='cpu')
        torch.cuda.set_rng_state(cuda_rng_state)

    start_epoch = checkpoint['epoch']
    hyperparams = checkpoint.get('hyperparams', {})

    print(f"Loaded checkpoint from {load_path}, resume from epoch {start_epoch}")
    return start_epoch, hyperparams, optimizer, scheduler


def collate_graphs(x, tokenizer, max_length, max_label_length):
    node_to_sample_ptr = [0]
    edge_to_sample_ptr = [0]
    current = 0
    current_edge = 0
    batch_size = len(x)
    input_tensors = []
    input_attn_masks = []
    graph_tensors = []
    graph_attn_masks = []
    flat_gat_edges = []
    labels = []

    sub_method_masks = []

    for idx in range(batch_size):
        item = x[idx]
        root = item["root"]
        method_full_name = root["method_full_name"]
        method_body = root["method_body"]

        pos_end = method_full_name.find("(")
        pos = -1
        label = method_full_name
        if pos_end != -1:
            pos = method_full_name[:pos_end].rfind('.')
        if pos != -1 and pos_end != -1:
            label = method_full_name[pos + 1: pos_end]

        masked_root_method = method_body
        nodes, edges, num_son, childs = parse(root, 0, [[masked_root_method, 0]], [], 0, 1, 0, [])
        nodes = [item[0] for item in nodes]
        nodes = nodes[1:]

        current += len(nodes)
        node_to_sample_ptr.append(current)

        if model_path == "../CodeGen":
            input_text = f"{masked_root_method} ||| {label}"
            input_tensor = tokenizer(
                input_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )
            context_text = f"{masked_root_method} |||"

            context_encoded = tokenizer(
                context_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            context_len = torch.sum(context_encoded["attention_mask"]).item()

            tmp_input_ids = input_tensor["input_ids"]
            valid_start_idx = (tmp_input_ids != tokenizer.pad_token_id).nonzero()[0][-1].item() if (
                        tmp_input_ids != tokenizer.pad_token_id).any() else 0
            input_tensor["attention_mask"][0][:valid_start_idx] = 0
            input_tensor["attention_mask"][0][valid_start_idx:] = 1

            label_tensor = torch.full_like(tmp_input_ids, -100)
            valid_mask = (input_tensor["attention_mask"] == 1)
            if context_len < max_length:
                label_tensor[0, -(torch.sum(valid_mask).item() - context_len):] = \
                    tmp_input_ids[0, -(torch.sum(valid_mask).item() - context_len):]
            label_tensor[0, :valid_start_idx] = tmp_input_ids[0, :valid_start_idx]
        else:
            if model_path == "../PLBart":
                masked_root_method = "<java> " + masked_root_method
            input_tensor = tokenizer(
                masked_root_method,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

        mask = input_tensor["input_ids"] >= 32000
        indices = torch.nonzero(mask[0], as_tuple=True)[0]
        sub_method_mask = torch.zeros(seq_length, dtype=torch.bool)
        sub_method_mask[indices] = True
        sub_method_masks.append(sub_method_mask)

        input_tensors.extend(input_tensor["input_ids"])
        input_attn_masks.extend(input_tensor["attention_mask"])

        if len(nodes) != 0:
            graph_tensor = tokenizer(
                nodes,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt"
            )

            graph_tensors.extend(graph_tensor["input_ids"])
            graph_attn_masks.extend(graph_tensor["attention_mask"])

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

        if model_path == "../CodeGen":
            pass
        else:
            if model_path == "../PLBart":
                label = "<java> " + label
            label_tensor = tokenizer(
                label,
                truncation=True,
                padding="max_length",
                max_length=max_label_length,
                return_tensors="pt"
            ).input_ids

        labels.append(label_tensor)

    input_tensors = torch.stack(input_tensors, dim=0)
    input_attn_masks = torch.stack(input_attn_masks, dim=0)

    if len(graph_tensors) > 0:
        graph_tensors = torch.stack(graph_tensors, dim=0)
        graph_attn_masks = torch.stack(graph_attn_masks, dim=0)
    else:
        graph_tensors = None
        graph_attn_masks = None

    sub_method_masks = torch.stack(sub_method_masks, dim=0)

    node_to_sample_ptr = torch.tensor(node_to_sample_ptr, dtype=torch.long)
    edge_to_sample_ptr = torch.tensor(edge_to_sample_ptr, dtype=torch.long)
    labels = torch.stack(labels, dim=0).squeeze(1)
    gat_edges = flat_gat_edges

    return {"input_tensors": input_tensors, "label": labels,
            "input_attn_masks": input_attn_masks, "graph_tensors": graph_tensors,
            "graph_attn_masks": graph_attn_masks, "sub_method_masks": sub_method_masks,
            "gat_edges": gat_edges, "node_to_sample_ptr": node_to_sample_ptr, "edge_to_sample_ptr": edge_to_sample_ptr}


def parse(root, current_idx, nodes, edges, depth, threshold, num_son, childs):
    if "children" not in root.keys() or depth == threshold:
        return nodes, edges, num_son, childs
    children = root["children"]
    if len(children) > 0:
        for i in range(len(children)):
            method = children[i]["method_body"]

            num_son += 1
            nodes.append([method.strip(), num_son])
            childs.append(children[i])
            edges.append([current_idx, num_son])

            _, _, num_son, childs = parse(children[i], num_son, nodes, edges, depth + 1, threshold, num_son, childs)

    return nodes, edges, num_son, childs


set_seed()
tokenizer = AutoTokenizer.from_pretrained(model_path)

if model_path == "../CodeT5":
    model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)
elif model_path == "../CodeT5p-220m":
    model = T6ForConditionalGeneration.from_pretrained(model_path).to(device)
elif model_path == "../CodeGen":
    seq_length = 195
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = CodeGenForCausalLM.from_pretrained(model_path).to(device)
elif model_path == "../PLBart":
    seq_length = 165
    label_length = 25
    tokenizer = AutoTokenizer.from_pretrained(model_path, src_lang="java", tgt_lang="java", ignore_mismatched_sizes=True)
    model = PLBartForConditionalGeneration.from_pretrained(model_path).to(device)
else:
    exit(-1)
scaler = GradScaler(enabled=False)
trainable_params = [
    {
        "params": [param for name, param in model.named_parameters() if param.requires_grad and 'fusion' not in name],
        "lr": 3e-5,
        "weight_decay": 1e-3
    },
    {
        "params": [param for name, param in model.named_parameters() if param.requires_grad and 'fusion' in name],
        "lr": 3e-5,
        "weight_decay": 1e-3
    }
]

optimizer = AdamW(trainable_params)
set_seed(12345)
set_seed()
train_data = read_gsons(["train_d_1_b_128_l_16_no_test.json"])
train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)

dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length, label_length)
)


total_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

resume_training = False
print(f"total_steps: {total_steps}")
time.sleep(1)

start_epoch = 0
checkpoint_path = "/ckpt_epoch_2.pth"
if resume_training and os.path.exists(output_dir + checkpoint_path):
    start_epoch, hyperparams, optimizer, scheduler = load_checkpoint(output_dir + checkpoint_path, model, optimizer, scheduler)
    batch_size = hyperparams["batch_size"]
    seq_length = hyperparams['seq_length']
    label_length = hyperparams['label_length']
    trainable_params = hyperparams['trainable_params']
    warmup_steps = hyperparams['warmup_steps']
    max_nodes = hyperparams['max_nodes']

    train_dataset = CallTreeDataset(train_data, tokenizer=tokenizer, max_length=seq_length)
    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_graphs(x, tokenizer, seq_length,  label_length)
    )
    total_steps = len(dataloader) * num_epochs


model.train()
for epoch in range(start_epoch, num_epochs):
    total_loss = 0.0
    progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

    for batch in progress_bar:

        graph_ids = batch["graph_tensors"]
        input_ids = batch["input_tensors"].to(device)
        graph_attention_mask = batch["graph_attn_masks"]
        attention_mask = batch["input_attn_masks"].to(device)
        labels = batch["label"].to(device)
        sub_method_masks = batch["sub_method_masks"].to(device)
        gat_edges = [edge.to(device) for edge in batch["gat_edges"]]
        node_to_sample_ptr = batch["node_to_sample_ptr"].to(device)
        edge_to_sample_ptr = batch["edge_to_sample_ptr"].to(device)

        if graph_ids is not None:
            graph_ids = graph_ids.to(device)
            graph_attention_mask = graph_attention_mask.to(device)
        else:
            gat_edges = None
        #
        node_to_sample_ptr = torch.tensor([i for i in range(input_ids.shape[0] + 1)], device=device)
        gat_edges = None  # [edge.to(device) for edge in batch["gat_edges"]]
        # node_to_sample_ptr = None  # batch["node_to_sample_ptr"].to(device)
        # edge_to_sample_ptr = None  # batch["edge_to_sample_ptr"].to(device)

        model.zero_grad()
        data_type = torch.bfloat16
        if model_path == "../CodeGen":
            data_type = torch.float32
        with autocast(device_type=device.type, dtype=data_type):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                # gat_edges=gat_edges,
                # sub_method_masks=sub_method_masks,
                # node_to_sample_ptr=node_to_sample_ptr,
                # edge_to_sample_ptr=edge_to_sample_ptr,
                # graph_input_ids=graph_ids,
                # graph_attention_mask=graph_attention_mask,
            )
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"Batch Loss": loss.item(), "Node Size": graph_ids.shape[0] if graph_ids is not None else 0})

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1} avg_loss: {avg_loss:.4f}")
    time.sleep(0.1)

    # if epoch % 2 == 0 or epoch == num_epochs - 1:
    if (epoch + 1 > 0 and epoch % 1 == 0) or epoch == num_epochs - 1:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        checkpoint_path = output_dir + "/ckpt_" + f"epoch_{epoch + 1}" + ".pth"
        save_checkpoint(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            batch_size=batch_size,
            trainable_params=trainable_params,
            warmup_steps=warmup_steps,
            seq_length=seq_length,
            max_label_length=label_length,
            save_path=checkpoint_path
        )
        model.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)
        tokenizer.save_pretrained(f"{output_dir}/epoch_{epoch + 1}", safe_serialization=False)

print("training finish!")
