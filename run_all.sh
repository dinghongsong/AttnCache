#!/bin/bash

for k in {0..5}; do
  echo "======== Running with k = $k ========"

  # Step 1: Collect attention maps
  echo "Running collect_hs_apms_llama.py..."
  python collect_hs_apms_llama.py --ntrain "$k"

  # Step 2: Train FP model and build database
  echo "Running train_fp_and_build_db.py..."
  python train_fp_and_build_db.py --ntrain "$k"

  # Step 3: Test LLaMA model with AttnCache
  echo "Running test_llama.py..."
  python test_llama.py --ntrain "$k"

  echo "======== Completed k = $k ========"
  echo
done

echo "All steps for k = 0 to 5 completed."
