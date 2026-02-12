# Data Directory

This directory contains LiverTox data for processing.

## Structure

```
data/
├── sample/                    # Sample data for testing
│   └── sample_livertox.html  # Example LiverTox drug monograph
├── LiverTox_A_files/         # Category A drugs (high likelihood)
├── LiverTox_B_files/         # Category B drugs
├── LiverTox_C_files/         # Category C drugs  
├── LiverTox_D_files/         # Category D drugs
├── LiverTox_E_files/         # Category E drugs (unlikely)
├── LiverTox_DailyMed_files/  # DailyMed entries
├── LiverTox_HDS_files/       # Herbal and dietary supplements
└── LiverTox_Drugs_files/     # General drug entries
```

## Getting Full Data

The full LiverTox database is available from NIH:

1. Visit: https://www.ncbi.nlm.nih.gov/books/NBK547852/
2. Navigate to individual drug monographs
3. Save HTML files to appropriate category directories

**Note**: The full database contains ~1,400+ drug monographs and is not included in this repository due to size and copyright considerations.

## Sample Data

The `sample/` directory contains example files for:
- Testing the system
- Understanding the expected format
- Running basic functionality without full data

To test with sample data only:
```bash
# Move sample file to a category directory
mkdir -p LiverTox_A_files
cp sample/sample_livertox.html LiverTox_A_files/

# Run training
python main.py --train
```

## File Format

LiverTox HTML files should contain structured sections:
- `<section id="introduction">`
- `<section id="hepatotoxicity">`
- `<section id="mechanism_of_injury">`
- `<section id="outcome_and_management">`
- `<section id="references">`

The system automatically parses these sections and creates semantic chunks for retrieval.