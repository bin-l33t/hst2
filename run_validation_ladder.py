"""
Run the Full Action-Angle Validation Ladder

This script runs all validation tests in order of increasing complexity:
1. Rotor (trivial, no branch cuts)
2. SHO (analytic, with branch handling)
3. Loop integral (canonicity)
4. Timescale (Glinsky's claim)

Each step validates the machinery before moving to harder cases.
"""

import sys


def run_validation_ladder():
    """Run the full validation ladder"""

    print("\n" + "=" * 70)
    print("ACTION-ANGLE VALIDATION LADDER")
    print("=" * 70)
    print("\nThis ladder validates action-angle coordinate systems with")
    print("corrections to avoid false failures from branch cuts and")
    print("improper observation operators.")
    print()

    results = {}

    # Step 1: Utilities self-test
    print("\n>>> STEP 0: UTILITY FUNCTIONS <<<\n")
    try:
        from action_angle_utils import (
            angular_distance, wrap_to_2pi, unwrap_angle,
            safe_points_mask, circular_mean, circular_std
        )
        import numpy as np

        # Quick tests
        assert abs(angular_distance(0.1, 2*np.pi - 0.1) - 0.2) < 1e-10
        assert abs(wrap_to_2pi(2.5*np.pi) - 0.5*np.pi) < 1e-10
        print("  Utility functions: OK ✓")
        results['utilities'] = True
    except Exception as e:
        print(f"  Utility functions: FAILED - {e}")
        results['utilities'] = False
        return results

    # Step 1: Rotor
    print("\n>>> STEP 1: FREE ROTOR <<<\n")
    try:
        from test_rotor import test_rotor
        results['rotor'] = test_rotor()
    except Exception as e:
        print(f"  Rotor test failed with exception: {e}")
        results['rotor'] = False

    if not results['rotor']:
        print("\n  ⚠ Rotor failed - stopping ladder (fix trivial case first)")
        return results

    # Step 2: SHO
    print("\n>>> STEP 2: SIMPLE HARMONIC OSCILLATOR <<<\n")
    try:
        from test_sho_action_angle import test_sho
        results['sho'] = test_sho()
    except Exception as e:
        print(f"  SHO test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['sho'] = False

    if not results['sho']:
        print("\n  ⚠ SHO failed - stopping ladder (fix SHO before canonicity)")
        return results

    # Step 3: Canonicity
    print("\n>>> STEP 3: LOOP INTEGRAL (CANONICITY) <<<\n")
    try:
        from test_loop_integral import test_sho_canonicity
        results['canonicity'] = test_sho_canonicity()
    except Exception as e:
        print(f"  Canonicity test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['canonicity'] = False

    # Step 4: Timescale
    print("\n>>> STEP 4: TIMESCALE TESTS <<<\n")
    try:
        from test_timescale import (
            test_timescale_exact_observation,
            test_timescale_unknown_origin,
            test_timescale_with_phase_diffusion
        )

        # Control test first
        results['timescale_control'] = test_timescale_exact_observation()
        print()

        # Unknown origin test
        results['timescale_unknown'] = test_timescale_unknown_origin()
        print()

        # Phase diffusion test
        results['timescale_diffusion'] = test_timescale_with_phase_diffusion()

    except Exception as e:
        print(f"  Timescale test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results['timescale_control'] = False
        results['timescale_unknown'] = False
        results['timescale_diffusion'] = False

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION LADDER SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:25s}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  🎉 ALL TESTS PASSED - Action-angle framework validated!")
    else:
        print("  ⚠ Some tests failed - review output above")

    print("\n" + "=" * 70)

    return results


if __name__ == "__main__":
    results = run_validation_ladder()

    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)
