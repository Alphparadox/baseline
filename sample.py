import os
import time
import torch
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# --- Helper Download and Data Prep ---
presentation_type = "multi"  # "multi" or "single"
difficulty_type = "kiva"     # "kiva" or "kiva-adults"
output_folder = "./output"
prefix = "" if difficulty_type == "kiva" else "adults_"

os.makedirs(output_folder, exist_ok=True)

# NOTE: This script assumes your *edited* helper.py (with './' paths)
# is in the same directory.

print("Downloading images...")
os.system(f"wget -q 'https://storage.googleapis.com/kiva_test/{prefix}{presentation_type}_image.zip' -O '{prefix}{presentation_type}_image.zip'")
os.system(f"unzip -qo '{prefix}{presentation_type}_image.zip' -d ./")
os.system("rm -rf ./__MACOSX")

import helper
data_dict = helper.prepare_data(presentation_type)
print(f"Data ready. Total samples: {len(data_dict)}")

# --- Show Example ---
concept = "2DRotation"
helper.show_concept_example(data_dict, concept, presentation_type)
# These are prompt TEMPLATES (they have "{}" in them)
system_prompt, general_cross_rule_prompt, general_within_rule_prompt, extrapolation_prompt = helper.display_all_prompts(presentation_type)

# --- Load Local LLaVA 13B ---
print("Loading local LLaVA model (13B)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "llava-hf/llava-1.5-13b-hf"

# Automatically map model across GPUs (recommended for 13B)
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print(f"✅ Model loaded successfully on device(s): {model.device if hasattr(model, 'device') else 'auto-mapped'}")

# --- Inference Function ---
def run_llava(prompt, image_path=None, max_tokens=150):
    """Run LLaVA locally with image and text."""
    if image_path is not None:
        if isinstance(image_path, list):
            images = [Image.open(p).convert("RGB") for p in image_path]
        else:
            images = [Image.open(image_path).convert("RGB")]
    else:
        images = None

    inputs = processor(prompt, images=images, return_tensors="pt").to(model.device if hasattr(model, "device") else device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens)

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return decoded.strip()

# --- Run Evaluation ---
print("\n🚀 Starting evaluation loop...")
evaluation_output_folder = f"{output_folder}/{presentation_type}_{difficulty_type}/"
os.makedirs(evaluation_output_folder, exist_ok=True)
csv_path = os.path.join(evaluation_output_folder, "all_results.csv")

results_list = []
start_time = time.time()

# --- START OF MAJOR CORRECTION ---
# We must generate the ground truth and final prompts *inside* the loop.

for idx, (img_id, img_metadata) in enumerate(data_dict.items()):
    if idx % 25 == 0 and idx != 0:
        elapsed = time.time() - start_time
        print(f"Processed {idx}/{len(data_dict)} images ({elapsed:.1f}s elapsed)")

    img_info = data_dict[img_id]
    parts = img_id.split("_")
    transform = parts[0] if len(parts) > 0 else "Unknown"
    variation = parts[1] if len(parts) > 1 else "Unknown"
    
    try:
        # --- 1. Generate Cross-Domain Prompt & Ground Truth ---
        # helper.py generates the options and the correct answer, e.g., "(1)"
        cross_options, gt_cross_domain, correct_concept = helper.generate_cross_options(img_id)
        cross_options_str = ", ".join(cross_options)
        # We format the template prompt with the options
        final_cross_prompt = general_cross_rule_prompt.format(cross_options_str)

        # --- 2. Generate Within-Domain Prompt & Ground Truth ---
        within_options, gt_within_domain, correct_param = helper.generate_within_options(img_id, img_info)
        within_options_str = ", ".join(within_options)
        # We format the template prompt with the options
        final_within_prompt = general_within_rule_prompt.format(within_options_str)

        # --- 3. Get Extrapolation Ground Truth ---
        # The correct key from the JSON is 'correct', e.g., "(A)"
        gt_extrapolation = img_info['correct'] 

        # --- 4. Run Inference with correct paths AND correct prompts ---
        if presentation_type == "multi":
            train_id = '_'.join(img_id.split('_')[:2])
            train_path = f'./multi_image/{train_id}_train.jpg'
            
            # Use the FINAL formatted prompt
            ans_cross_domain = run_llava(final_cross_prompt, train_path)
            ans_within_domain = run_llava(final_within_prompt, train_path)
            
            test0_path = f'./multi_image/{img_id}_test_0.jpg'
            test1_path = f'./multi_image/{img_id}_test_1.jpg'
            test2_path = f'./multi_image/{img_id}_test_2.jpg'
            extrapolation_image_list = [train_path, test0_path, test1_path, test2_path]
            
            # Extrapolation prompt has no options to format
            ans_extrapolation = run_llava(extrapolation_prompt, extrapolation_image_list)

        else: # presentation_type == "single"
            image_path = f'./single_image/{img_id}_single.jpg'
            
            # Use the FINAL formatted prompts
            ans_cross_domain = run_llava(final_cross_prompt, image_path)
            ans_within_domain = run_llava(final_within_prompt, image_path)
            ans_extrapolation = run_llava(extrapolation_prompt, image_path)

        # --- END OF MAJOR CORRECTION ---

        # --- Debugging Step ---
        print(f"\n--- [Debug] Processing Image: {img_id} ---")
        print(f"  [Cross Domain]   Model Answer: {ans_cross_domain}")
        print(f"                   Ground Truth: {gt_cross_domain}")
        print(f"  [Within Domain]  Model Answer: {ans_within_domain}")
        print(f"                   Ground Truth: {gt_within_domain}")
        print(f"  [Extrapolation]  Model Answer: {ans_extrapolation}")
        print(f"                   Ground Truth: {gt_extrapolation}")
        # --- End Debug ---

    except Exception as e:
        print(f"⚠️ Error processing {img_id}: {e}")
        ans_cross_domain = ans_within_domain = ans_extrapolation = "ERROR"

    results_list.append({
        "img_id": img_id,
        "transform": transform,
        "variation": variation,
        "cross_domain_answer": ans_cross_domain,
        "cross_domain_ground_truth": gt_cross_domain,
        "within_domain_answer": ans_within_domain,
        "within_domain_ground_truth": gt_within_domain,
        "extrapolation_answer": ans_extrapolation,
        "extrapolation_ground_truth": gt_extrapolation,
    })

    # Checkpoint save every 50 images
    if (idx + 1) % 50 == 0:
        print(f"💾 Saving checkpoint to {csv_path}...")
        pd.DataFrame(results_list).to_csv(csv_path, index=False)

# Final save
pd.DataFrame(results_list).to_csv(csv_path, index=False)
print(f"\n✅ Evaluation complete. Results saved to {csv_path}")

# --- Analysis ---
print("\n📊 Starting Analysis ---")
analysis_output_folder = f"{output_folder}/{presentation_type}_{difficulty_type}/"

if not os.path.exists(os.path.join(analysis_output_folder, "all_results.csv")):
    print(f"⚠️ Missing results file in {analysis_output_folder}")
else:
    print(f"Analyzing results from: {analysis_output_folder}")
    answer_types = ["cross_domain", "within_domain", "extrapolation"]
    transform_types = ["2DRotation", "Colour", "Counting", "Reflect", "Resize"]

    analysis_results = helper.init_analysis_results(answer_types, transform_types)
    csv_correctness_dict = helper.load_correctness_from_csv(analysis_output_folder)
    analysis_results = helper.update_analysis_results(analysis_results, data_dict, csv_correctness_dict, answer_types)
    analysis_results = helper.compute_accuracy_by_unique_trial(analysis_results)
    analysis_results = helper.aggregate_by_transformation_category(analysis_results)

    print("\n--- Subcategory Results ---")
    helper.print_subcategory_results(analysis_results)
    print("\n--- Plotting Results ---")
    # This was plot_results(analysis_results, evaluation_output_folder)
    # But helper.py's plot_results takes 1 argument. 
    # This might be another bug, but we will follow helper.py's definition.
    # If plotting fails, we may need to edit helper.py
    try:
        helper.plot_results(analysis_results, evaluation_output_folder)
        print(f"📈 Plots saved in {evaluation_output_folder}")
    except TypeError:
        print("Attempting plot fix (1 arg)...")
        try:
            # Try the other function signature from the helper.py you sent
            helper.plot_results(analysis_results) 
            print(f"📈 Plots saved in {evaluation_output_folder}")
        except Exception as e:
            print(f"⚠️ Plotting failed: {e}")
            
print("\n✅ Script Finished Successfully.")