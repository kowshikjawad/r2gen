import argparse
from modules.tokenizers import Tokenizer
from modules.datasets import BaseDataset

class MockArgs:
    def __init__(self, ann_path, image_dir='data/iu_xray/images/'):
        self.ann_path = ann_path
        self.image_dir = image_dir
        self.max_seq_length = 60
        self.threshold = 1
        self.dataset_name = 'iu_xray'

def verify_layman_dataset(ann_path):
    print(f"Verifying dataset loading from: {ann_path}")
    args = MockArgs(ann_path)
    tokenizer = Tokenizer(args)
    
    dataset = BaseDataset(args, tokenizer, split='train')
    print(f"Successfully loaded dataset! Total train samples: {len(dataset)}")
    print(f"Vocabulary size: {len(tokenizer.idx2token)}")
    
    first_example = dataset.examples[0]
    print("\nSample processed dataset record:")
    print(f"  ID: {first_example['id']}")
    print(f"  Report: {first_example['report']}")
    print(f"  Token IDs: {first_example['ids'][:10]}...")
    
    decoded = tokenizer.decode(first_example['ids'])
    print(f"  Decoded text: {decoded}")
    print("\nVerification PASSED.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation_layman.json')
    args = parser.parse_args()
    verify_layman_dataset(args.ann_path)
