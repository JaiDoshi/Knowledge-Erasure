import torch

def compute_kl(pretrained_model, current_model, batch, device):

    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    loss = -(prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()

    return loss


def get_answer_loss(operation, batch, model, device="cuda:0"):

    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        losses.append(position_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss