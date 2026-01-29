#!/bin/bash
#
# Deploy Experiment 1.1 to nigel and run Phase 1
#

set -e

echo "======================================"
echo "Deploying Experiment 1.1 to nigel"
echo "======================================"

# 1. Copy files to nigel
echo ""
echo "1. Copying files to nigel..."

scp phase1_feature_discovery.py vincent@nigel.birs.ca:~/sae_cluster_probe/
scp causal_intervention.py vincent@nigel.birs.ca:~/sae_cluster_probe/
scp EXPERIMENT_1.1_README.md vincent@nigel.birs.ca:~/sae_cluster_probe/

echo "   Files copied successfully"

# 2. Check data availability
echo ""
echo "2. Checking data availability on nigel..."

ssh vincent@nigel.birs.ca << 'EOF'
echo "   Checking for datasets..."

if [ -f ~/lightbright/data/gold_106.json ]; then
    echo "   ✓ gold_106.json found"
else
    echo "   ✗ gold_106.json NOT found"
fi

if [ -f ~/af-detection-benchmark/data/classes/compliant_hard_negative.json ]; then
    echo "   ✓ compliant_hard_negative.json found"
else
    echo "   ✗ compliant_hard_negative.json NOT found"
    echo "   Checking alternative locations..."
    find ~ -name "compliant_hard_negative.json" 2>/dev/null | head -3
fi
EOF

# 3. Setup environment
echo ""
echo "3. Setting up environment on nigel..."

ssh vincent@nigel.birs.ca << 'EOF'
cd ~/sae_cluster_probe

# Check if venv exists
if [ ! -d ~/venv ]; then
    echo "   Creating virtual environment..."
    python3 -m venv ~/venv
fi

# Activate and install dependencies
source ~/venv/bin/activate

echo "   Installing dependencies..."
pip install -q torch transformers huggingface_hub safetensors peft tqdm numpy

echo "   Environment ready"
EOF

echo ""
echo "======================================"
echo "Deployment complete!"
echo "======================================"
echo ""
echo "To run Phase 1 experiment:"
echo ""
echo "  ssh vincent@nigel.birs.ca"
echo "  cd ~/sae_cluster_probe"
echo "  source ~/venv/bin/activate"
echo ""
echo "  # Quick test (single layer, small sample)"
echo "  python phase1_feature_discovery.py --layers 40 --n_af 10 --n_hard_neg 10"
echo ""
echo "  # Full experiment (all layers)"
echo "  python phase1_feature_discovery.py --layers 20 30 40 50 --n_af 50 --n_hard_neg 100"
echo ""
echo "  # Check results"
echo "  cat phase1_results/phase1_results.json | jq '.layers.layer_40.top_features[:5]'"
echo ""
