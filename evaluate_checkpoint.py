import torch
import argparse
import numpy as np
import os
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from models.r2gen import R2GenModel
from main import parse_agrs

def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create validation and test data loaders (skipping train_dataloader to save memory/time)
    print("Loading data loaders...")
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    print("Building model architecture...")
    model = R2GenModel(args, tokenizer)

    # Prepare device
    device = torch.device('cuda:0' if args.n_gpu > 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # load checkpoint
    checkpoint_path = args.resume
    if not checkpoint_path:
        # Check defaults
        default_path = os.path.join(args.save_dir, 'model_best.pth')
        if os.path.exists(default_path):
            checkpoint_path = default_path
        else:
            raise ValueError("Please specify the checkpoint path using the --resume flag, or ensure model_best.pth exists in the save directory.")

    print(f"Loading checkpoint from: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded checkpoint trained up to epoch: {epoch}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded raw state dict checkpoint.")

    model.eval()

    # Evaluate on Validation set
    print("\nEvaluating on validation set (Full)...")
    with torch.no_grad():
        val_gts, val_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(val_dataloader):
            images = images.to(device)
            output = model(images, mode='sample')
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            val_res.extend(reports)
            val_gts.extend(ground_truths)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_dataloader)} batches...")
        
        print("Computing validation scores...")
        val_met = compute_scores({i: [gt] for i, gt in enumerate(val_gts)},
                                 {i: [re] for i, re in enumerate(val_res)})
        print("\n=== Validation Results ===")
        for k, v in val_met.items():
            print(f"  {k:10s}: {v:.4f}")

    # Evaluate on Test set
    print("\nEvaluating on test set (Full)...")
    with torch.no_grad():
        test_gts, test_res = [], []
        for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):
            images = images.to(device)
            output = model(images, mode='sample')
            reports = model.tokenizer.decode_batch(output.cpu().numpy())
            ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
            test_res.extend(reports)
            test_gts.extend(ground_truths)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_dataloader)} batches...")
        
        print("Computing test scores...")
        test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                  {i: [re] for i, re in enumerate(test_res)})
        print("\n=== Test Results ===")
        for k, v in test_met.items():
            print(f"  {k:10s}: {v:.4f}")

if __name__ == '__main__':
    main()
