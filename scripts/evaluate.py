from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

def evaluate_bleu(model, dataloader, vocab, idx_to_word, beam_size=3, max_len=20, num_samples=100, use_beam=True):
    model.eval()
    scores = []
    smooth = SmoothingFunction().method1

    actual_samples = min(len(dataloader), num_samples)

    with torch.no_grad():
        for i, (images, captions) in enumerate(dataloader):
            if i >= actual_samples:
                break

            for img, refs in zip(images, captions):
                img = img.to(next(model.parameters()).device)

                if use_beam:
                    pred_ids = model.generate_caption_beam(img, vocab, beam_size=beam_size, max_len=max_len)
                else:
                    pred_ids = model.generate_caption_greedy(img, vocab, max_len=max_len)

                pred_tokens = [
                    idx_to_word[i] for i in pred_ids
                    if i not in {vocab['<start>'], vocab['<end>'], vocab['<pad>']}
                ]

                ref_tokens = [
                    idx_to_word[i.item()] for i in refs
                    if i.item() not in {vocab['<start>'], vocab['<end>'], vocab['<pad>']}
                ]

                # Compute BLEU-1 (unigram)
                score = sentence_bleu(
                    [ref_tokens],
                    pred_tokens,
                    weights=(1.0, 0, 0, 0),
                    smoothing_function=smooth
                )
                scores.append(score)

    avg_bleu = sum(scores) / len(scores)
    return avg_bleu
