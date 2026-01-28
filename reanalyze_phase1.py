#!/usr/bin/env python3
"""
Reanalyze existing Phase 1 results with corrected metrics.

Check if AF-specific features exist in the data we already collected.
"""

import json
import numpy as np
from pathlib import Path

def reanalyze_results(results_path: str):
    """Reanalyze with correct AF-specific filtering."""

    with open(results_path) as f:
        results = json.load(f)

    print("=" * 80)
    print("Reanalyzing Phase 1 Results")
    print("=" * 80)

    for layer_key in results['layers'].keys():
        layer_data = results['layers'][layer_key]
        layer = layer_data['layer']

        print(f"\nLayer {layer}:")

        # Count by direction
        af_specific = [f for f in layer_data['top_features'] if f['differential'] > 0 and f['cohens_d'] > 0.5]
        hn_specific = [f for f in layer_data['top_features'] if f['differential'] < 0 and f['cohens_d'] < -0.5]

        print(f"  Top 100 features breakdown:")
        print(f"    AF-specific (fires MORE on AF): {len(af_specific)}")
        print(f"    HN-specific (fires MORE on hard neg): {len(hn_specific)}")

        if af_specific:
            print(f"\n  Top 10 AF-specific features:")
            for i, feat in enumerate(af_specific[:10], 1):
                print(f"    {i}. Feature {feat['feature_id']:5d}: "
                      f"AF={feat['af_mean']:7.2f}, "
                      f"HN={feat['hard_neg_mean']:7.2f}, "
                      f"diff=+{feat['differential']:7.2f}, "
                      f"d={feat['cohens_d']:5.2f}")
        else:
            print("\n  ❌ No AF-specific features in top 100!")

        if hn_specific:
            print(f"\n  Top 5 HN-specific (vocabulary) features:")
            for i, feat in enumerate(hn_specific[:5], 1):
                print(f"    {i}. Feature {feat['feature_id']:5d}: "
                      f"AF={feat['af_mean']:7.2f}, "
                      f"HN={feat['hard_neg_mean']:7.2f}, "
                      f"diff={feat['differential']:7.2f}, "
                      f"d={feat['cohens_d']:5.2f}")

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Overall assessment
    total_af_specific = sum(
        len([f for f in results['layers'][lk]['top_features']
             if f['differential'] > 0 and f['cohens_d'] > 0.5])
        for lk in results['layers'].keys()
    )

    if total_af_specific >= 10:
        print(f"\n✅ SUCCESS: {total_af_specific} AF-specific features found in top 100!")
        print("   These fire MORE on AF samples than hard negatives")
        print("   Potential intent-encoding features")
    else:
        print(f"\n❌ FAILURE: Only {total_af_specific} AF-specific features in top 100")
        print("   Most discriminative features are vocabulary detectors")
        print("   (They fire on hard negatives with RLHF keywords, not on AF)")

    return results

if __name__ == "__main__":
    import sys

    results_path = sys.argv[1] if len(sys.argv) > 1 else "./phase1_results/phase1_results.json"

    if Path(results_path).exists():
        reanalyze_results(results_path)
    else:
        print(f"ERROR: {results_path} not found")
        print("\nTry:")
        print("  python reanalyze_phase1.py phase1_results/phase1_results.json")
