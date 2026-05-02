# Raw Datasets

This directory stores raw battery aging datasets. **These files are gitignored** due to their size.

## Download Instructions

### 1. NASA PCoE Battery Aging Dataset (Primary)
- **URL:** https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
- **File:** `BatteryAgingARC.zip` (contains B0005, B0006, B0007, B0018 .mat files)
- **Size:** ~150 MB
- **Format:** MATLAB .mat
- **Download & extract:**
  ```bash
  # Download and place in this directory
  unzip BatteryAgingARC.zip -d nasa_pcoe/
  ```

### 2. Oxford Battery Degradation Dataset (Cross-validation)
- **URL:** https://howey.eng.ox.ac.uk/data-and-code/
- **Format:** MATLAB .mat
- **Download & extract:**
  ```bash
  mkdir oxford/
  # Place downloaded files in oxford/
  ```

### 3. CALCE Battery Research Group (Cross-validation)
- **URL:** https://calce.umd.edu/battery-data
- **Format:** CSV / Excel
- **Download & extract:**
  ```bash
  mkdir calce/
  # Place downloaded files in calce/
  ```

## Expected Directory Structure After Download

```
ml/data/raw/
├── README.md          (this file)
├── nasa_pcoe/
│   ├── B0005.mat
│   ├── B0006.mat
│   ├── B0007.mat
│   └── B0018.mat
├── oxford/
│   └── ...
└── calce/
    └── ...
```
