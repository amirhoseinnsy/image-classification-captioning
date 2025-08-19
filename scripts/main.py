import torch
from torch.utils.data import  dataloader
from data.data_loader import *
from scripts.train import *
from models.model import *
from utils.visualization import *

def filter_captions(image_dict, caption_dict):
        return {k: v for k, v in caption_dict.items() if k in image_dict}

def modelParametersCount(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def ImageCaptioning():
    train_list, cv_list, test_list = image_data_loader()
    pic_to_cap, pic_to_ids, vocab = build_tokenizer_and_vocab(text_data_loader())
    
    train_data = Flickr8(train_list, filter_captions(train_list, pic_to_ids))
    cv_data = Flickr8(cv_list, filter_captions(cv_list, pic_to_ids))

    train_data = dataloader.DataLoader(train_data, batch_size=60, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>']))
    cv_data = dataloader.DataLoader(cv_data, batch_size=60, shuffle=True, collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>']))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove_matrix = load_glove_embeddings(vocab)
    model = ImageCaptioningModel(len(vocab), glove_matrix)
    ckpt_path = r"D:\Deep\Assignment 3\models\saved_models\model_38.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # If you saved only the state_dict:
    if isinstance(checkpoint, dict) and not any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        # e.g. {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # checkpoint IS the state_dict
        model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    idx_to_word = {idx: word for word, idx in vocab.items()}

    # train(
    #     model=model,
    #     train_data=train_data,
    #     val_data=cv_data, 
    #     device=device,
    #     optimizer=optimizer,
    #     epochs=100,
    #     save_epoch=1,
    #     patience=5
    # )
    
    
    test_data = Flickr8(test_list, filter_captions(test_list, pic_to_ids))
    test_loader = dataloader.DataLoader(
        test_data, batch_size=60, shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>'])
    )

    avg_test_bleu = evaluate_bleu(
        model=model,
        dataloader=test_loader,
        vocab=vocab,
        idx_to_word=idx_to_word,
        beam_size=3,
        max_len=20,
        num_samples=len(test_loader), 
        use_beam=True
    )
    print(f"\n*** Test set BLEU-1 (unigram) = {avg_test_bleu:.4f} ***")


def ImageCaptioningFinal():
    train_list, cv_list, test_list = image_data_loader()
    pic_to_cap, pic_to_ids, vocab = build_tokenizer_and_vocab(text_data_loader())

    train_data = Flickr8(train_list, filter_captions(train_list, pic_to_ids))
    cv_data = Flickr8(cv_list, filter_captions(cv_list, pic_to_ids))

    train_loader = dataloader.DataLoader(
        train_data, batch_size=60, shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>'])
    )
    cv_loader = dataloader.DataLoader(
        cv_data, batch_size=60, shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>'])
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    glove_matrix = load_glove_embeddings(vocab)
    model = ImageCaptioningModel(len(vocab), glove_matrix, use_attention=True)
    ckpt_path = r"D:\Deep\Assignment 3\models\saved_models\model_final_1.pth"
    checkpoint = torch.load(ckpt_path, map_location=device)

    # If you saved only the state_dict:
    if isinstance(checkpoint, dict) and not any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        # e.g. {'model_state_dict': ..., 'optimizer_state_dict': ..., ...}
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # checkpoint IS the state_dict
        model.load_state_dict(checkpoint)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

    idx_to_word = {idx: word for word, idx in vocab.items()}

    # train_with_attention_bleu_beam(
    #     model=model,
    #     train_data=train_loader,
    #     val_data=cv_loader,
    #     device=device,
    #     optimizer=optimizer,
    #     epochs=100,
    #     save_epoch=1,
    #     patience=5,
    #     teacher_forcing_ratio=1.0,
    #     vocab=vocab,
    #     idx_to_word=idx_to_word,
    #     use_beam=True,
    #     beam_size=3
    # )



    test_data = Flickr8(test_list, filter_captions(test_list, pic_to_ids))
    test_loader = dataloader.DataLoader(
        test_data, batch_size=60, shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx=vocab['<pad>'])
    )

        # call on one example
    # --- visualize attention on one test image ---
    idx_to_word = {idx: word for word, idx in vocab.items()}
    visualize_attention(
        image_path="D:\Deep\Assignment 3\data\dataset\Images/19212715_20476497a3.jpg",
        model=model,
        transform=test_transform,
        vocab=vocab,
        idx_to_word=idx_to_word,
        max_len=20,
        n_cols=6
    )

    # avg_test_bleu = evaluate_bleu(
    #     model=model,
    #     dataloader=test_loader,
    #     vocab=vocab,
    #     idx_to_word=idx_to_word,
    #     beam_size=3,
    #     max_len=20,
    #     num_samples=len(test_loader), 
    #     use_beam=True
    # )
    # print(f"\n*** Test set BLEU-1 (unigram) = {avg_test_bleu:.4f} ***")


if __name__ == "__main__":
    ImageCaptioningFinal()