#!/bin/bash

# Step 1: Collect attention maps
echo "Running collect_hs_apms_llama.py..."
python collect_hs_apms_llama.py

# Step 2: Train FP model and build database
echo "Running train_fp_and_build_db.py..."
python train_fp_and_build_db.py

# Step 3: Test LLaMA model with AttnCache
echo "Running test_llama.py..."
python test_llama.py

echo "All steps completed."


# args.device = torch.device('cuda')
#     args.device = torch.device('cpu')  0.470