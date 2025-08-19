import torch
from tqdm import tqdm
from utils.metrics import metrics
import logging
from datetime import datetime
import yaml
from scripts.evaluate import evaluate_bleu

log_data = {"logs": []}

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    log_data["logs"].append(log_entry)
    logging.info(message)
    with open("debug_log.txt", "a") as dbg:
        dbg.write(log_entry + "\n")


log_file = "config/logging_ImageCaptioning_Final.yaml"

def train(model, train_data, val_data, device, optimizer, epochs, save_epoch=1, patience=5, teacher_forcing_ratio=0.0):
    global log_data
    model = model.to(device)
    train_acc = metrics()
    loss_avg_train = metrics()
    val_acc = metrics()
    loss_avg_val = metrics()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')
    epochs_no_improve = 0

    start = datetime.now()
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        train_acc.reset()
        loss_avg_train.reset()
        loops = tqdm(enumerate(train_data, 1), total=len(train_data), leave=True)

        for batch_index, (images, captions) in loops:
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()

            outputs = model(images, captions, teacher_forcing_ratio=teacher_forcing_ratio)  # [B, T-1, vocab_size]
            targets = captions[:, 1:]  

            output_len = outputs.size(1)
            target_len = targets.size(1)
            if output_len > target_len:
                outputs = outputs[:, :target_len, :]
            elif output_len < target_len:
                targets = targets[:, :output_len]

            outputs = outputs.reshape(-1, outputs.size(-1))  # [B*T, vocab]
            targets = targets.reshape(-1)  # [B*T]

            loss_ = loss_fn(outputs, targets)
            loss_.backward()
            optimizer.step()

            with torch.no_grad():
                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float()
                mask = (targets != 0)
                acc = acc[mask].sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
            train_acc.step(acc.item(), images.size(0))
            loss_avg_train.step(loss_.item(), images.size(0))
            loops.set_description(f'Epoch {epoch}/{epochs}')
            loops.set_postfix(
                batch_loss=loss_.item(),
                average_loss=loss_avg_train.average,
                accuracy=train_acc.average,
            )

        model.eval()
        val_acc.reset()
        loss_avg_val.reset()
        with torch.inference_mode():
            loops = tqdm(enumerate(val_data, 1), total=len(val_data), leave=True)
            for batch_index, (images, captions) in loops:
                images = images.to(device)
                captions = captions.to(device)

                outputs = model(images, captions, teacher_forcing_ratio=0.0)  # no teacher forcing in eval
                targets = captions[:, 1:]

                output_len = outputs.size(1)
                target_len = targets.size(1)
                if output_len > target_len:
                    outputs = outputs[:, :target_len, :]
                elif output_len < target_len:
                    targets = targets[:, :output_len]

                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)

                loss_ = loss_fn(outputs, targets)

                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float()
                mask = (targets != 0)
                acc = acc[mask].sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
                val_acc.step(acc.item(), images.size(0))
                loss_avg_val.step(loss_.item(), images.size(0))
                loops.set_description(f'Epoch {epoch}/{epochs}')
                loops.set_postfix(
                    batch_loss=loss_.item(),
                    average_loss=loss_avg_val.average,
                    accuracy=val_acc.average,
                )

        train_acc.history()
        loss_avg_train.history()
        val_acc.history()
        loss_avg_val.history()
        message = (f'Epoch {epoch}/{epochs} | Train Loss: {loss_avg_train.average:.4f} | '
                   f'Train Accuracy: {train_acc.average:.4f} | '
                   f'Val Loss: {loss_avg_val.average:.4f} | '
                   f'Val Accuracy: {val_acc.average:.4f}')
        log_message(message)
        with open(log_file, 'a') as f:
            yaml.dump(log_data, f)

        if loss_avg_val.average < best_val_loss:
            best_val_loss = loss_avg_val.average
            torch.save(model.state_dict(), f'models/saved_models/model_{epoch}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_message("Early stopping triggered.")
                break

    end = datetime.now()
    log_message(f'Training Time: {str(end - start)}')
    with open(log_file, 'a') as f:
        yaml.dump(log_data, f)

    return model, train_acc.hist, loss_avg_train.hist, val_acc.hist, loss_avg_val.hist


def train_with_attention_bleu_beam(
    model, train_data, val_data, device, optimizer, epochs, save_epoch=1, patience=5,
    teacher_forcing_ratio=1.0, vocab=None, idx_to_word=None, use_beam=True, beam_size=3):

    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
    best_bleu = 0
    epochs_no_improve = 0

    train_acc = metrics()
    loss_avg_train = metrics()
    val_acc = metrics()
    loss_avg_val = metrics()
    bleu_metric = metrics()

    start = datetime.now()
    for epoch in range(1, epochs + 1):
        model.train()
        train_acc.reset()
        loss_avg_train.reset()
        loops = tqdm(enumerate(train_data, 1), total=len(train_data), leave=True)

        for batch_index, (images, captions) in loops:
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()

            outputs = model(images, captions, teacher_forcing_ratio=teacher_forcing_ratio)
            targets = captions[:, 1:]

            if outputs.size(1) > targets.size(1):
                outputs = outputs[:, :targets.size(1), :]
            elif outputs.size(1) < targets.size(1):
                targets = targets[:, :outputs.size(1)]

            outputs = outputs.reshape(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss_ = loss_fn(outputs, targets)
            loss_.backward()
            optimizer.step()

            with torch.no_grad():
                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float()
                mask = (targets != 0)
                acc = acc[mask].sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)
            train_acc.step(acc.item(), images.size(0))
            loss_avg_train.step(loss_.item(), images.size(0))
            loops.set_description(f'Epoch {epoch}/{epochs}')
            loops.set_postfix(
                train_loss=loss_.item(),
                avg_train_loss=loss_avg_train.average,
                train_acc=train_acc.average
            )

        model.eval()
        val_acc.reset()
        loss_avg_val.reset()
        with torch.inference_mode():
            val_loops = tqdm(enumerate(val_data, 1), total=len(val_data), leave=True)
            for batch_index, (images, captions) in val_loops:
                images = images.to(device)
                captions = captions.to(device)
                outputs = model(images, captions, teacher_forcing_ratio)
                targets = captions[:, 1:]

                if outputs.size(1) > targets.size(1):
                    outputs = outputs[:, :targets.size(1), :]
                elif outputs.size(1) < targets.size(1):
                    targets = targets[:, :outputs.size(1)]

                outputs = outputs.reshape(-1, outputs.size(-1))
                targets = targets.reshape(-1)

                loss_ = torch.nn.functional.cross_entropy(outputs, targets, ignore_index=0)
                preds = outputs.argmax(dim=-1)
                acc = (preds == targets).float()
                mask = (targets != 0)
                acc = acc[mask].sum() / mask.sum() if mask.sum() > 0 else torch.tensor(0.0)

                val_acc.step(acc.item(), images.size(0))
                loss_avg_val.step(loss_.item(), images.size(0))

                val_loops.set_description(f'Validation Epoch {epoch}')
                val_loops.set_postfix(
                    val_loss=loss_.item(),
                    avg_val_loss=loss_avg_val.average,
                    val_acc=val_acc.average
                )


        avg_bleu = evaluate_bleu(
            model=model,
            dataloader=val_data,
            vocab=vocab,
            idx_to_word=idx_to_word,
            num_samples=500,
            beam_size=beam_size,
            max_len=20,
            use_beam=use_beam
        )
        bleu_metric.step(avg_bleu, 1)
        bleu_metric.history()

        message = (
            f'Epoch {epoch}/{epochs} | Train Loss: {loss_avg_train.average:.4f} | '
            f'Train Acc: {train_acc.average:.4f} | Val Loss: {loss_avg_val.average:.4f} | '
            f'Val Acc: {val_acc.average:.4f} | AVG BLEU: {avg_bleu:.4f} | Best BLEU: {best_bleu}'
        )
        log_message(message)

        if avg_bleu > best_bleu:
            best_bleu = avg_bleu
            torch.save(model.state_dict(), f'models/saved_models/model_Final_{epoch}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log_message("Early stopping triggered by BLEU.")
                break

        with open(log_file, 'a') as f:
            yaml.dump(log_data, f)

        teacher_forcing_ratio = max(0.5, 1.0 - epoch * 0.1)

    end = datetime.now()
    log_message(f'Training Time: {str(end - start)}')
    with open(log_file, 'a') as f:
        yaml.dump(log_data, f)

    return model, bleu_metric.hist, loss_avg_train.hist, val_acc.hist, loss_avg_val.hist


def test_bleu(model, test_loader, vocab, device, beam_size=3):
    """
    Run inference on the test set and compute corpus BLEU.
    - model: your image-captioning model
    - test_loader: DataLoader yielding (images, captions, lengths)
    - vocab: vocabulary with idx2word mapping and SOS/EOS/PAD tokens
    - device: 'cuda' or 'cpu'
    - beam_size: beam width for decoding (use 1 for greedy)
    """
    model.eval()
    all_refs = []
    all_hyps = []

    with torch.no_grad():
        for images, captions, _ in test_loader:
            images = images.to(device)
            # generate with beam search (or greedy if beam_size=1)
            # assume your model has a .sample() method that returns list of lists of idxs
            hyps = model.sample(images, vocab, beam_size=beam_size)

            for ref_idxs, hyp_idxs in zip(captions, hyps):
                # convert reference to word list (drop special tokens)
                ref_words = [
                    vocab.idx2word[idx.item()]
                    for idx in ref_idxs
                    if idx.item() not in {vocab.SOS, vocab.EOS, vocab.PAD}
                ]
                # convert hypothesis likewise
                hyp_words = [
                    vocab.idx2word[idx]
                    for idx in hyp_idxs
                    if idx not in {vocab.SOS, vocab.EOS, vocab.PAD}
                ]
                all_refs.append([ref_words])  # note nesting for corpus_bleu
                all_hyps.append(hyp_words)
