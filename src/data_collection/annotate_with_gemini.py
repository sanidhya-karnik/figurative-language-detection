import os
import time
import pandas as pd
import google.generativeai as genai
import json
from tqdm import tqdm
import typing_extensions as typing

# === CONFIGURATION ===
INPUT_FILE = 'data/raw/news_articles.csv'
OUTPUT_FILE = 'data/processed/news_annotations_gemini.csv'
MODEL_NAME = 'gemini-1.5-flash'  # fast and cost-effective
# MODEL_NAME = 'gemini-1.5-pro' # Use if higher quality is needed (more expensive/slower)

def setup_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")
        print("Please set it in your terminal: $env:GEMINI_API_KEY='your_key_here'")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(MODEL_NAME)

def get_batch_annotation_prompt(texts):
    prompt = """
Analyze the following list of texts for figurative language (Metaphor and Irony/Sarcasm).

Definitions:
1. Metaphor: Words used with non-literal meanings (e.g., "drowning in work", "climbing back up"). Follow MIPVU broadly: does the contextual meaning contrast with the basic meaning?
2. Irony/Sarcasm: The intended meaning contradicts the surface meaning (e.g., saying "Great weather" during a storm). Look for context-dependent incongruity.
3. Literal: The text is not figurative language.
Return a JSON array of objects, one for each text in the same order. Each object must have:
{
  "is_metaphor": boolean,
  "is_irony": boolean,
  "is_literal": boolean,
  "reasoning": "string explanation"
}

Texts to analyze:
"""
    for idx, text in enumerate(texts):
        prompt += f"\n--- TEXT {idx+1} ---\n{text}\n"
    
    return prompt

def annotate_with_gemini():
    model = setup_gemini()
    if not model:
        return

    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    try:
        df = pd.read_csv(INPUT_FILE)
        print(f"Loaded {len(df)} reviews from {INPUT_FILE}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # 2. Check for existing progress
    if os.path.exists(OUTPUT_FILE):
        print(f"Found existing annotation file: {OUTPUT_FILE}")
        annotated_df = pd.read_csv(OUTPUT_FILE)
        start_index = len(annotated_df)
        print(f"Resuming from index {start_index}...")
    else:
        annotated_df = pd.DataFrame(columns=[
            'text', 'label_metaphor', 'label_irony', 'reasoning', 
            'source_file_index', 'original_sentiment', 'source'
        ])
        start_index = 0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 3. Annotation Loop
    BATCH_SIZE = 10  # Number of texts to send in one API call
    
    print(f"Starting batch annotation with {MODEL_NAME} (Batch size: {BATCH_SIZE})...")
    
    try:
        # Process in batches
        for i in tqdm(range(start_index, len(df), BATCH_SIZE), initial=start_index, total=len(df)):
            batch_df = df.iloc[i:i+BATCH_SIZE]
            batch_texts = []
            valid_indices = [] # Keep track of original indices for this batch

            # Prepare batch
            for idx, row in batch_df.iterrows():
                text = row.get('text') or row.get('review') or row.get('body') or row.get('content') or row.get('review_body')
                if not isinstance(text, str) or not text.strip():
                    continue
                # Truncate very long texts
                if len(text) > 2000: 
                    text = text[:2000] + "..."
                batch_texts.append(text)
                valid_indices.append(idx)
            
            if not batch_texts:
                continue

            try:
                # Call Gemini API
                prompt = get_batch_annotation_prompt(batch_texts)
                response = model.generate_content(prompt, generation_config={"response_mime_type": "application/json"})
                
                # Parse JSON response
                try:
                    results = json.loads(response.text)
                    
                    if not isinstance(results, list):
                         # Handle case where model returns single object instead of list
                        if isinstance(results, dict): 
                            results = [results]
                        else:
                            raise ValueError("API response is not a list")

                    # Match results to valid_indices
                    # Note: We rely on the model maintaining order. If lengths mismatch, we might lose data for this batch.
                    if len(results) != len(batch_texts):
                        print(f"Warning: Batch {i}: Sent {len(batch_texts)} texts but got {len(results)} results. Skipping batch.")
                        continue

                    batch_annotations = []
                    for local_idx, res in enumerate(results):
                        original_idx = valid_indices[local_idx]
                        row = df.iloc[original_idx]
                        
                        batch_annotations.append({
                            'text': batch_texts[local_idx],
                            'label_metaphor': 1 if res.get('is_metaphor') else 0,
                            'label_irony': 1 if res.get('is_irony') else 0,
                            'reasoning': res.get('reasoning', ''),
                            'source_file_index': original_idx,
                            'original_sentiment': row.get('sentiment_label') or row.get('sentiment') or row.get('rating'),
                            'source': 'IMDb'
                        })

                    # Save this batch immediately
                    new_df = pd.DataFrame(batch_annotations)
                    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
                        new_df.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                    else:
                        new_df.to_csv(OUTPUT_FILE, index=False)

                except json.JSONDecodeError:
                    print(f"JSON Error at batch {i}. Raw response: {response.text}")
                
                # Rate limit handling
                time.sleep(2) 

            except Exception as e:
                print(f"API Error at batch starting index {i}: {e}")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\nAnnotation interrupted by user.")

    print("Done.")

if __name__ == "__main__":
    annotate_with_gemini()

