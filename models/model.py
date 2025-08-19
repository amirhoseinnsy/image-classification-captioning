import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

class AdditiveAttention(nn.Module):
    def __init__(self, image_dim, hidden_dim, attn_dim=256):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, attn_dim)
        self.hidden_proj = nn.Linear(hidden_dim, attn_dim)
        self.score_proj = nn.Linear(attn_dim, 1)

    def forward(self, image_feats, hidden_state):
        img = self.image_proj(image_feats)                        # [B, L, A]
        hid = self.hidden_proj(hidden_state).unsqueeze(1)         # [B, 1, A]
        e = self.score_proj(torch.tanh(img + hid)).squeeze(-1)    # [B, L]
        alpha = F.softmax(e, dim=1)                               # [B, L]
        context = torch.bmm(alpha.unsqueeze(1), image_feats).squeeze(1)  # [B, D]
        return context, alpha

def load_glove_embeddings(vocab, glove_path="D:\\Deep\\Assignment 3\\models\\glove.6B/glove.6b.300d.txt", embedding_dim=300):
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = vec

    embedding_matrix = np.zeros((len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
    return torch.tensor(embedding_matrix, dtype=torch.float)

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, glove_matrix, embed_dim=300, hidden_dim=512, num_layers=1, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # CNN Encoder
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.encoder_cnn = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))
        self.feature_proj = nn.Linear(2048, hidden_dim)

        if use_attention:
            self.attention = AdditiveAttention(image_dim=hidden_dim, hidden_dim=hidden_dim)

        self.embedding = nn.Embedding.from_pretrained(glove_matrix, freeze=True)
        self.embedding_proj = nn.Linear(embed_dim, hidden_dim)

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_dim * 2 if use_attention else hidden_dim, vocab_size)

    def caption_with_attention(self, image, vocab, max_len=20):
        self.eval()
        device = next(self.parameters()).device

        # 1) Encode image
        img = image.to(device).unsqueeze(0)  # (1,3,H,W)
        with torch.no_grad():
            feats = self.encoder_cnn(img)
        feats = self.adaptive_pool(feats)              # (1,2048,14,14)
        feats = feats.flatten(2).permute(0, 2, 1)       # (1, L, 2048)
        feats = self.feature_proj(feats)                # (1, L, hidden_dim)

        # 2) Init state
        context = feats.mean(dim=1)                     # (1, hidden_dim)
        inputs = context.unsqueeze(1)                   # (1, 1, hidden_dim)
        hidden = None

        # 3) Start token
        token = torch.LongTensor([vocab['<start>']]).to(device)
        seq, alphas = [vocab['<start>']], []

        # 4) Decode loop
        for _ in range(max_len):
            emb = self.embedding(token)                # (1, embed_dim)
            emb = self.embedding_proj(emb)             # (1, hidden_dim)

            # ✳ Ensure emb is (1, 1, hidden_dim)
            if emb.dim() == 1:
                emb = emb.unsqueeze(0).unsqueeze(0)
            elif emb.dim() == 2:
                emb = emb.unsqueeze(1)

            # ✳ Ensure inputs is (1, seq_len, hidden_dim)
            if inputs.dim() == 2:
                inputs = inputs.unsqueeze(1)

            # ✳ Concatenate along time dimension
            inputs = torch.cat([inputs, emb], dim=1)    # (1, t+1, hidden_dim)
            output, hidden = self.lstm(inputs, hidden)
            h_t = output[:, -1, :]                      # (1, hidden_dim)

            if self.use_attention:
                ctx, alpha = self.attention(feats, h_t)
                combined = torch.cat([h_t, ctx], dim=1) # (1, hidden_dim*2)
                logits = self.fc(combined)
                alphas.append(alpha.squeeze(0).cpu())
            else:
                logits = self.fc(h_t)

            token = logits.argmax(dim=-1).squeeze(0)
            seq.append(token.item())
            if token.item() == vocab['<end>']:
                break

        return seq, alphas

    def forward(self, images, captions, teacher_forcing_ratio=0.0):
        batch_size, max_len = captions.size()
        with torch.no_grad():
            features = self.encoder_cnn(images)
        features = self.adaptive_pool(features).flatten(2).permute(0, 2, 1)  # [B, L, 2048] -> [B, L, H]
        features = self.feature_proj(features)

        context = features.mean(dim=1)
        inputs = context.unsqueeze(1)
        outputs = []
        input_token = captions[:, 0]
        hidden = None

        for t in range(1, max_len):
            emb = self.embedding(input_token).unsqueeze(1)
            emb = self.embedding_proj(emb)

            lstm_input = torch.cat([inputs, emb], dim=1)
            lstm_out, hidden = self.lstm(lstm_input, hidden)
            h_t = lstm_out[:, -1, :]

            if self.use_attention:
                context_vec, _ = self.attention(features, h_t)
                combined = torch.cat([h_t, context_vec], dim=1)
                out_t = self.fc(combined)
            else:
                out_t = self.fc(h_t)

            outputs.append(out_t.unsqueeze(1))

            if self.training and torch.rand(1).item() > teacher_forcing_ratio:
                input_token = out_t.argmax(dim=-1)
            else:
                input_token = captions[:, t]

            inputs = lstm_out

        return torch.cat(outputs, dim=1)

    def generate_caption_beam(self, image, vocab, beam_size=3, max_len=20):
        self.eval()
        with torch.no_grad():
            features = self.encoder_cnn(image.unsqueeze(0))
            features = self.adaptive_pool(features).flatten(2).permute(0, 2, 1)
            features = self.feature_proj(features)
            context = features.mean(dim=1)

            sequences = [[[], 0.0, context.unsqueeze(1), None]]

            for _ in range(max_len):
                all_candidates = []
                for seq, score, inputs, hidden in sequences:
                    if seq and seq[-1] == vocab['<end>']:
                        all_candidates.append((seq, score, inputs, hidden))
                        continue

                    input_token = torch.tensor([[seq[-1]]] if seq else [[vocab['<start>']]], device=image.device)
                    emb = self.embedding(input_token)
                    emb = self.embedding_proj(emb)
                    

                    lstm_input = torch.cat([inputs, emb], dim=1)
                    output, hidden = self.lstm(lstm_input, hidden)
                    h_t = output[:, -1, :]

                    if self.use_attention:
                        context_vec, _ = self.attention(features, h_t)
                        combined = torch.cat([h_t, context_vec], dim=1)
                        output = self.fc(combined)
                    else:
                        output = self.fc(h_t)

                    log_probs = torch.log_softmax(output, dim=-1).squeeze(0)
                    topk = torch.topk(log_probs, beam_size)

                    for i in range(beam_size):
                        word_idx = topk.indices[i].item()
                        word_score = topk.values[i].item()
                        candidate = (seq + [word_idx], score + word_score, lstm_input, hidden)
                        all_candidates.append(candidate)

                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            return sequences[0][0]

    def generate_caption_greedy(self, image, vocab, max_len=20):
        self.eval()
        with torch.no_grad():
            features = self.encoder_cnn(image.unsqueeze(0))
            features = self.adaptive_pool(features).flatten(2).permute(0, 2, 1)
            features = self.feature_proj(features)
            context = features.mean(dim=1)
            inputs = context.unsqueeze(1)

            caption = [vocab['<start>']]
            hidden = None

            for _ in range(max_len):
                input_token = torch.tensor([[caption[-1]]], device=image.device)
                emb = self.embedding(input_token)
                emb = self.embedding_proj(emb)

                inputs = torch.cat([inputs, emb], dim=1)
                output, hidden = self.lstm(inputs, hidden)
                h_t = output[:, -1, :]

                if self.use_attention:
                    context_vec, _ = self.attention(features, h_t)
                    combined = torch.cat([h_t, context_vec], dim=1)
                    output = self.fc(combined)
                else:
                    output = self.fc(h_t)

                next_word = output.argmax(dim=-1).item()
                caption.append(next_word)
                if next_word == vocab['<end>']:
                    break

            return caption
