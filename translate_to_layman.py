import os
import json
import argparse
import urllib.request
import urllib.error

# High-accuracy fallback dictionary for standard radiology phrases
FALLBACK_DICTIONARY = {
    "the lungs are clear . no pleural effusion or pneumothorax .": 
        "the lungs look completely clear and healthy with no fluid buildup around the lungs and no collapsed lung .",
    "heart size is normal . lungs are clear .": 
        "the heart size is normal and the lungs look clear and healthy .",
    "no acute cardiopulmonary disease .": 
        "no active or immediate heart or lung disease is visible .",
    "mild cardiomegaly without acute pulmonary findings .": 
        "the heart is slightly enlarged , but there are no urgent or active lung issues ."
}


def call_gemini_api(report, api_key):
    """Translates a medical report string to accurate layman English using Gemini REST API with retry handling."""
    import time
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    prompt = (
        "You are an expert medical communicator. Translate the following formal radiology report into "
        "accurate, patient-friendly, plain-English layman language. Preserve all clinical findings (normal and abnormal) "
        "with 100% accuracy, but replace medical jargon with simple language (e.g., 'pleural effusion' -> 'fluid buildup around lungs', "
        "'cardiomegaly' -> 'enlarged heart', 'pneumothorax' -> 'collapsed lung'). Return ONLY the translated sentence, lowercased, "
        "with space-separated punctuation matching standard caption format.\n\n"
        f"Original Report: {report}\n\nLayman Translation:"
    )
    
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')

def call_gemini_api(report, api_key):
    """Translates a medical report string to accurate layman English using Gemini REST API."""
    import time
    models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash"]
    
    prompt = (
        "You are an expert medical communicator. Translate the following formal radiology report into "
        "accurate, patient-friendly, plain-English layman language. Preserve all clinical findings (normal and abnormal) "
        "with 100% accuracy, but replace medical jargon with simple language (e.g., 'pleural effusion' -> 'fluid buildup around lungs', "
        "'cardiomegaly' -> 'enlarged heart', 'pneumothorax' -> 'collapsed lung'). Return ONLY the translated sentence, lowercased, "
        "with space-separated punctuation matching standard caption format.\n\n"
        f"Original Report: {report}\n\nLayman Translation:"
    )
    
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ]
    }
    
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(payload).encode('utf-8')

    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                res_body = response.read().decode('utf-8')
                res_json = json.loads(res_body)
                translated_text = res_json['candidates'][0]['content']['parts'][0]['text'].strip()
                return translated_text.lower()
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Quota / Rate limit exceeded - switch immediately to smart fallback
                return None
            elif e.code == 404:
                continue
            else:
                return None
        except Exception:
            return None
            
    return None


def translate_report(report, cache, api_key=None):
    """Returns the layman translation for a report, checking cache first then API/fallback."""
    if report in cache:
        return cache[report]
    
    translated = None
    if api_key:
        translated = call_gemini_api(report, api_key)
    
    if not translated:
        # Smart rule-based jargon replacement fallback
        t = report
        replacements = [
            ("pleural effusion", "fluid buildup around lungs"),
            ("pleural effusions", "fluid buildup around lungs"),
            ("pneumothorax", "collapsed lung"),
            ("cardiomegaly", "enlarged heart"),
            ("atelectasis", "partial lung collapse"),
            ("granuloma", "small benign lung scar"),
            ("granulomas", "small benign lung scars"),
            ("emphysema", "chronic lung damage"),
            ("consolidation", "lung infection or inflammation"),
            ("opacities", "cloudy spots"),
            ("opacity", "cloudy spot"),
            ("pulmonary vascularity", "lung blood vessels"),
            ("cardiomediastinal silhouette", "heart and chest shape"),
            ("mediastinal contours", "chest structure outlines"),
            ("osseous structures", "bones"),
            ("within normal limits", "completely normal"),
            ("unremarkable", "normal"),
            ("acute pulmonary findings", "urgent lung issues"),
            ("acute cardiopulmonary disease", "active heart or lung disease")
        ]
        for jargon, simple in replacements:
            t = t.replace(jargon, simple)
            t = t.replace(jargon.capitalize(), simple)
        translated = t
            
    cache[report] = translated
    return translated


def main():
    parser = argparse.ArgumentParser(description="Translate medical report annotations to layman language.")
    parser.add_argument('--input_ann', type=str, default='data/iu_xray/annotation.json', help='Input annotation JSON path')
    parser.add_argument('--output_ann', type=str, default='data/iu_xray/annotation_layman.json', help='Output annotation JSON path')
    parser.add_argument('--cache_file', type=str, default='data/translation_cache.json', help='Translation cache JSON path')
    parser.add_argument('--api_key', type=str, default=None, help='Gemini API key (or set GEMINI_API_KEY env var)')
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('GEMINI_API_KEY')
    
    if not os.path.exists(args.input_ann):
        raise FileNotFoundError(f"Input annotation file not found: {args.input_ann}")

    print(f"Loading input annotations from: {args.input_ann}")
    with open(args.input_ann, 'r') as f:
        data = json.load(f)

    # Load cache if exists
    cache = {}
    if os.path.exists(args.cache_file):
        with open(args.cache_file, 'r') as f:
            try:
                cache = json.load(f)
                print(f"Loaded {len(cache)} existing cached translations from {args.cache_file}")
            except Exception:
                cache = {}

    # Count total examples
    total_dataset_items = sum(len(data[s]) for s in ['train', 'val', 'test'] if s in data)

    total_examples = 0
    translated_data = {}

    count = 0
    for split in ['train', 'val', 'test']:
        if split not in data:
            continue
        translated_data[split] = []
        for item in data[split]:
            total_examples += 1
            orig_report = item['report']
            layman_report = translate_report(orig_report, cache, api_key=api_key)
            
            new_item = dict(item)
            new_item['report'] = layman_report
            new_item['orig_report'] = orig_report  # Keep original for reference
            translated_data[split].append(new_item)

            count += 1
            if count % 10 == 0 or count == total_dataset_items:
                print(f"Progress: [{count}/{total_dataset_items}] reports processed | {len(cache)} unique translations in cache", flush=True)
                os.makedirs(os.path.dirname(args.cache_file) or '.', exist_ok=True)
                with open(args.cache_file, 'w') as f:
                    json.dump(cache, f, indent=4)

    # Save updated cache
    os.makedirs(os.path.dirname(args.cache_file) or '.', exist_ok=True)
    with open(args.cache_file, 'w') as f:
        json.dump(cache, f, indent=4)

    # Save new annotation file
    os.makedirs(os.path.dirname(args.output_ann) or '.', exist_ok=True)
    with open(args.output_ann, 'w') as f:
        json.dump(translated_data, f, indent=4)

    print("\n--- Translation Summary ---")
    print(f"Total reports processed: {total_examples}")
    print(f"Unique reports cached: {len(cache)}")
    print(f"Saved new layman annotation file to: {args.output_ann}")
    
    # Print sample comparison
    print("\nSample Comparisons:")
    first_split = list(translated_data.keys())[0]
    for sample in translated_data[first_split][:3]:
        print(f"  [ID {sample['id']}]")
        print(f"    Original: {sample['orig_report']}")
        print(f"    Layman:   {sample['report']}")


if __name__ == '__main__':
    main()
