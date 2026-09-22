# Data Dictionary — `house_prices.csv`

**Rows:** 300 · **Columns:** 8 · **Missing values:** none in this sample

| Column | Type | Description | Example |
|---|---|---|---|
| `Property_ID` | string | Unique listing identifier. Not used as a model feature. | `PROP0001` |
| `Area` | integer | Floor area of the property, in square feet. | `3712` |
| `Bedrooms` | integer | Number of bedrooms. | `4` |
| `Bathrooms` | integer | Number of bathrooms. | `3` |
| `Age` | integer | Age of the property in years. | `36` |
| `Location` | categorical | One of `City Center`, `Suburb`, `Rural`. | `Rural` |
| `Property_Type` | categorical | One of `House`, `Apartment`, `Villa`. | `House` |
| `Price` | integer | Sale price in INR (₹). **Target variable.** | `22260000` |

## Engineered features (added by `src/data_preprocessing.py`)

| Feature | Definition | Rationale |
|---|---|---|
| `total_rooms` | `Bedrooms + Bathrooms` | Captures overall size independent of area |
| `bath_bed_ratio` | `Bathrooms / Bedrooms` (0 if no bedrooms) | Higher-end properties often have more bathrooms per bedroom |
| `is_new` | 1 if `Age <= 5`, else 0 | Flags a "new build" price premium |
| `area_per_room` | `Area / total_rooms` | Spaciousness per room, independent of raw size |
| `age_bucket` | `Age` binned into `0-5`, `6-15`, `16-30`, `30+` | Lets tree models split on age non-linearly |

## Known data quality rules enforced by the pipeline

- Duplicate rows are dropped.
- Rows with non-positive `Area` or `Price`, or `Age` outside `[0, 150]`, are dropped as physically impossible.
- Rows with a `Location` or `Property_Type` value outside the sets above are dropped rather than guessed.
- Missing numeric values (if present in future data) are imputed with the column median.
