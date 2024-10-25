import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datasets import load_dataset
from metrics import AlignmentMetrics  # Ensure this is correctly implemented/imported
import itertools
import numpy as np
import gc
import logging
from multiprocessing import Pool, cpu_count, set_start_method, get_start_method
from functools import partial

# -------------------- Setup Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------- Configuration --------------------

# Generate all checkpoints first
all_checkpoints = sorted(
    {f"step{step}" for step in [0, 1000] + list(range(8000, 15000, 7000)) +
     list(range(15000, 143000, 7000)) + [143000]},
    key=lambda x: int(x.replace("step", ""))
)

# Select around 40 evenly spaced checkpoints
num_checkpoints = 40
selected_indices = np.linspace(0, len(all_checkpoints) - 1, num_checkpoints, dtype=int)
selected_checkpoints = [all_checkpoints[i] for i in selected_indices]

logger.info(f"Selected Checkpoints: {selected_checkpoints}")

# Define models and their checkpoints
model_checkpoints = {
    "pythia-14m": selected_checkpoints,
    "pythia-70m": selected_checkpoints,
    "pythia-160m": selected_checkpoints,
    "pythia-410m": selected_checkpoints,
    "pythia-1b": selected_checkpoints,
    "pythia-1.4b": selected_checkpoints,
}

# Directory to store extracted features
FEATURES_DIR = "model_features"
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load a sentence classification dataset (e.g., SST-2 from GLUE)
logger.info("Loading dataset...")
dataset = load_dataset("glue", "sst2", split="validation[:1000]")  # Using 1000 samples for faster processing
sentences = dataset["sentence"]
logger.info(f"Loaded {len(sentences)} sentences.")

# -------------------- Feature Extraction Functions --------------------

def get_dynamic_batch_size(model_name):
    """Adjust batch size based on model size to prevent OOM errors."""
    if "1b" in model_name or "1.4b" in model_name:
        return 16  # Reduced from 32
    elif "410m" in model_name:
        return 32  # Reduced from 64
    else:
        return 64  # Reduced from 128

def extract_features(model, tokenizer, sentences, batch_size=64):
    """Extract last hidden state features from the model for given sentences using mixed precision."""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    features = []
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()), torch.no_grad():
        for i in tqdm(range(0, len(sentences), batch_size), desc="Extracting features", leave=False):
            batch_sentences = sentences[i:i + batch_size]
            inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Set the model to output hidden states
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]
            cls_tokens = last_hidden_state[:, 0, :]  # Assuming CLS token is at position 0
            features.append(cls_tokens.cpu())
            torch.cuda.empty_cache()  # Clear cache to free up memory

    features = torch.cat(features, dim=0)
    return features

def extract_features_for_checkpoint(args):
    """Extract features for a given model at a specific checkpoint."""
    model_name, checkpoint, sentences = args
    feature_save_path = os.path.join(FEATURES_DIR, f"{model_name.replace('/', '_')}_{checkpoint}.pt")

    # If features are already saved, load them
    if os.path.exists(feature_save_path):
        logger.info(f"Loading saved features for {model_name} at {checkpoint}...")
        try:
            features = torch.load(feature_save_path)
            return (model_name, checkpoint, features)
        except Exception as e:
            logger.error(f"Failed to load saved features for {model_name} at {checkpoint}: {e}")
            return (model_name, checkpoint, None)

    try:
        logger.info(f"Loading model {model_name} at checkpoint {checkpoint}...")
        model = AutoModelForCausalLM.from_pretrained(
            f"EleutherAI/{model_name}", 
            revision=checkpoint, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}", revision=checkpoint)
        
        # Ensure the tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        batch_size = get_dynamic_batch_size(model_name)
        logger.info(f"Extracting features with batch size {batch_size}...")
        features = extract_features(model, tokenizer, sentences, batch_size)
        
        # Save the extracted features to disk
        torch.save(features, feature_save_path)
        logger.info(f"Saved features for {model_name} at {checkpoint} to {feature_save_path}")
    except Exception as e:
        logger.error(f"Failed to process {model_name} at {checkpoint}: {e}")
        features = None
    finally:
        # Clean up to free memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
    
    return (model_name, checkpoint, features)

# -------------------- Alignment Metric Computation --------------------

def compute_alignment(task):
    """Compute alignment metric for a single task."""
    combination, checkpoint, path_a, path_b = task
    if os.path.exists(path_a) and os.path.exists(path_b):
        try:
            features_a = torch.load(path_a)
            features_b = torch.load(path_b)

            score = AlignmentMetrics.mutual_knn(features_a, features_b, topk=50)

            return {'Combination': combination, 'Checkpoint': checkpoint, 'Score': score}
        except Exception as e:
            logger.error(f"Error calculating alignment for {combination} at {checkpoint}: {e}")
            return None
    else:
        logger.warning(f"Missing features for {combination} at {checkpoint}. Skipping.")
        return None

def compute_alignment_metrics(model_checkpoints, selected_checkpoints):
    """Compute alignment metrics between all unique unordered pairs of models at each selected checkpoint."""
    alignment_records = []
    model_names = sorted(model_checkpoints.keys())  # Sort model names to ensure consistent ordering

    # Generate all unique unordered model pairs
    model_pairs = list(itertools.combinations(model_names, 2))
    total_combinations = len(model_pairs) * len(selected_checkpoints)
    logger.info(f"Computing alignment metrics for {len(model_pairs)} model pairs across {len(selected_checkpoints)} checkpoints (Total: {total_combinations} combinations).")

    # Prepare alignment tasks
    tasks = []
    for checkpoint in selected_checkpoints:
        for model_a, model_b in model_pairs:
            combination = f"{model_a.split('-')[-1]}-{model_b.split('-')[-1]}"  # e.g., "14m-70m"
            feature_path_a = os.path.join(FEATURES_DIR, f"{model_a.replace('/', '_')}_{checkpoint}.pt")
            feature_path_b = os.path.join(FEATURES_DIR, f"{model_b.replace('/', '_')}_{checkpoint}.pt")
            tasks.append((combination, checkpoint, feature_path_a, feature_path_b))

    # Determine number of parallel workers
    num_workers = min(cpu_count(), 8)  # Adjust based on your machine's capabilities

    logger.info(f"Using {num_workers} parallel workers for alignment metric computation.")
    with Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap_unordered(compute_alignment, tasks), total=len(tasks), desc="Alignment Metrics"):
            if result is not None:
                alignment_records.append(result)

    score_df = pd.DataFrame(alignment_records)

    # Verify the number of expected unique pairs
    num_models = len(model_names)
    num_pairs = num_models * (num_models - 1) // 2
    expected_pairs = num_pairs * len(selected_checkpoints)
    actual_pairs = len(score_df)
    if actual_pairs < expected_pairs:
        logger.warning(f"Expected {expected_pairs} alignment scores, but got {actual_pairs}. Some pairs may be missing.")
    else:
        logger.info(f"All {actual_pairs} alignment scores computed successfully.")

    return score_df

# -------------------- Plotting the Results --------------------

def plot_alignment_scores(score_df):
    """Plot alignment scores as training progresses."""
    
    # Ensure that each (Combination, Checkpoint) is unique
    score_df = score_df.drop_duplicates(subset=['Combination', 'Checkpoint'])
    
    # Create a 'step' column by extracting numeric value from 'Checkpoint'
    if 'step' not in score_df.columns:
        score_df['step'] = score_df['Checkpoint'].str.replace('step', '', regex=False).astype(int)
    
    # Sort the DataFrame by 'step' to ensure chronological order
    score_df_sorted = score_df.sort_values(by='step').reset_index(drop=True)
    
    # Set the visual style for the plot
    sns.set(style="whitegrid")
    
    # Initialize the matplotlib figure with a specified size
    plt.figure(figsize=(20, 12))
    
    # Create a line plot with markers for better visibility
    sns.lineplot(
        data=score_df_sorted,
        x='step',
        y='Score',
        hue='Combination',
        marker='o',
        palette='tab20',  # Expanded palette to accommodate more combinations
        legend='full'
    )
    
    # Add a title and labels with increased font sizes for better readability
    plt.title('Alignment Score (mutual_knn) vs. Training Step per Model Combination', fontsize=20)
    plt.xlabel('Training Step', fontsize=16)
    plt.ylabel('Alignment Score', fontsize=16)
    
    # Improve x-axis labels by rotating them if there are many steps
    plt.xticks(score_df_sorted['step'].unique(), rotation=45)
    
    # Adjust legend to fit outside the plot
    plt.legend(title='Model Combination', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent clipping of labels and ensure everything fits well
    plt.tight_layout()
    
    # Display the plot
    plt.show()
    logger.info("Plotting completed.")

# -------------------- Main Execution --------------------

def main():
    # Step 1: Extract features for all models and checkpoints
    process_models_sequentially(model_checkpoints, sentences)
    
    # Step 2: Compute alignment metrics
    score_df = compute_alignment_metrics(model_checkpoints, selected_checkpoints)
    
    # Step 3: Save the alignment scores for future reference
    scores_save_path = os.path.join(FEATURES_DIR, "alignment_scores_scaling.csv")
    score_df.to_csv(scores_save_path, index=False)
    logger.info(f"Saved alignment scores to {scores_save_path}")
    
    # Step 4: Plot the results
    plot_alignment_scores(score_df)
    
    # Step 5: (Optional) Save the plot as an image file
    # Uncomment the following lines if you wish to save the plot
    # plt.savefig(os.path.join(FEATURES_DIR, "mean_alignment_score_per_combination_vs_step.png"), dpi=300)
    # logger.info("Plot saved successfully.")

def process_models_sequentially(model_checkpoints, sentences):
    """Process each model and its checkpoints sequentially to extract features."""
    total_tasks = sum(len(checkpoints) for checkpoints in model_checkpoints.values())
    logger.info(f"Starting feature extraction for {len(model_checkpoints)} models across {total_tasks} checkpoints.")

    # Prepare arguments for multiprocessing
    tasks = []
    for model_name, checkpoints in model_checkpoints.items():
        for checkpoint in checkpoints:
            tasks.append((model_name, checkpoint, sentences))

    # Determine the number of parallel workers (limited to CPU cores)
    num_workers = min(cpu_count(), 4)  # Adjust based on your machine's capabilities

    logger.info(f"Using {num_workers} parallel workers for feature extraction.")
    with Pool(processes=num_workers, initializer=None) as pool:
        for model_name, checkpoint, features in tqdm(pool.imap_unordered(extract_features_for_checkpoint, tasks), total=len(tasks), desc="Feature Extraction"):
            if features is not None:
                # Features are already saved to disk
                continue
            else:
                logger.warning(f"Features for {model_name} at {checkpoint} could not be extracted and will be skipped.")

    logger.info("Feature extraction completed for all models and checkpoints.")

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn'
    try:
        set_start_method('spawn')
    except RuntimeError:
        # Start method has already been set
        pass

    main()
