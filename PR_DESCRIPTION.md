# Pull Request: Implement in-frame complex variation (MNP + INDEL) handling

## Summary
This PR implements the missing functionality for handling in-frame complex variations that involve both multiple nucleotide polymorphisms (MNPs) and insertions/deletions (INDELs) in the variant effect prediction system.

## Problem Solved
Resolves the TODO comment in `_get_within_cds_effect()` function in `malariagen_data/veff.py` that was previously returning a placeholder effect "TODO in-frame complex variation (MNP + INDEL)" with "UNKNOWN" impact.

## Implementation Details

### Changes Made
- **File**: `malariagen_data/veff.py`
- **Function**: `_get_within_cds_effect()`
- **Lines**: ~25 lines of new implementation logic

### Logic Implemented
The new implementation handles complex variants where:
1. Both ref and alt sequences are >1 nucleotide
2. Lengths are different (not simple MNPs)  
3. Length difference is a multiple of 3 (in-frame)

**Classification Outcomes:**
- `CODON_CHANGE_PLUS_INDEL` - when amino acid changes are detected between ref and alt
- `CODON_INSERTION` - for pure insertions without codon changes (alt > ref)
- `CODON_DELETION` - for pure deletions without codon changes (ref > alt)
- All classified with "MODERATE" impact

### Testing
- **Added**: `tests/test_veff.py` with comprehensive test coverage
- **Test Scenarios**:
  - Complex variation with codon changes
  - Pure in-frame insertion without codon changes
  - Pure in-frame deletion without codon changes
- **Result**: All tests pass successfully

## Code Quality
- Follows existing code style and patterns
- Includes clear comments explaining the logic
- Maintains consistency with other effect classifications in the same function
- Proper error handling and edge case consideration

## Impact
This improvement enhances the accuracy of variant effect predictions for complex genomic variations, which is crucial for:
- More precise genomic analysis
- Better understanding of complex mutations
- Improved downstream analysis pipelines

## GSoC Context
This contribution is submitted as part of the GSoC 2026 application process for the malariagen-data-python project. It demonstrates:
- Understanding of genomic data processing
- Ability to work with complex bioinformatics algorithms
- Commitment to code quality and testing
- Familiarity with open source development practices

## Testing Instructions
To verify the implementation:
```bash
python -m pytest tests/test_veff.py -v
```

## Review Checklist
- [x] Code follows project style guidelines
- [x] Tests are included and passing
- [x] Documentation is updated where necessary
- [x] Implementation handles edge cases appropriately
- [x] No breaking changes to existing functionality

## Files Changed
- `malariagen_data/veff.py` - Main implementation
- `tests/test_veff.py` - Test coverage

## Commit Hash
`5550789f6acd0edc8e1a61c08902422e0d1a677a`
