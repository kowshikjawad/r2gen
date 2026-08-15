import torch
import argparse
import os
from PIL import Image
from torchvision import transforms
from modules.tokenizers import Tokenizer
from models.r2gen import R2GenModel

def parse_args():
    parser = argparse.ArgumentParser()

    # Inference specific arguments
    parser.add_argument('--image_paths', type=str, nargs='+', required=True, 
                        help='Path(s) to the chest X-ray image(s). (Provide 2 images for iu_xray, or 1 for mimic_cxr)')

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args = parser.parse_args()
    return args

def main():
    # parse arguments
    args = parse_args()
    
    # Create tokenizer
    tokenizer = Tokenizer(args)

    # Build model architecture
    print("Building model architecture...")
    model = R2GenModel(args, tokenizer)

    # Prepare device
    device = torch.device('cuda:0' if args.n_gpu > 0 and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)

    # Load checkpoint
    checkpoint_path = args.resume
    if not checkpoint_path:
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
    else:
        model.load_state_dict(checkpoint)
    print("Checkpoint loaded successfully.")

    model.eval()

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load and process the images
    processed_images = []
    print(f"Loading and preprocessing image(s): {args.image_paths}")
    for path in args.image_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found at path: {path}")
        img = Image.open(path).convert('RGB')
        img_tensor = transform(img)
        processed_images.append(img_tensor)

    # Stack images appropriately depending on the dataset mode
    if args.dataset_name == 'iu_xray':
        if len(processed_images) < 2:
            raise ValueError("IU X-Ray model requires 2 images (e.g. frontal and lateral views). Please provide 2 paths in --image_paths.")
        elif len(processed_images) > 2:
            print("Warning: More than 2 images provided. Only the first two will be used.")
        # Stack two images for the study
        image_input = torch.stack((processed_images[0], processed_images[1]), 0).unsqueeze(0)
    else:
        # Single image for MIMIC-CXR
        image_input = processed_images[0].unsqueeze(0)

    image_input = image_input.to(device)

    # Generate report
    print("\nGenerating report...")
    with torch.no_grad():
        output = model(image_input, mode='sample')
        generated_report = model.tokenizer.decode_batch(output.cpu().numpy())[0]

    print("\n" + "="*50)
    print("GENERATED RADIOLOGY REPORT:")
    print("="*50)
    print(generated_report)
    print("="*50 + "\n")

if __name__ == '__main__':
    main()
