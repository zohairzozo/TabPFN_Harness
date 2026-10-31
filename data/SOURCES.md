# Data sources

Every external dataset: where it came from, when, under which licence, and the SHA-256 of the file as used. The files themselves are not in git (`data/raw/` is ignored).

## fingrid_2019-01-01_5d.csv

- **What:** a slice of 5 consecutive days of 1-second grid frequency, 432,000 rows,
  days 2019-01-01 to 2019-01-05 (naive local time). Used as the split `real_control`.
- **Source:** Kruse, Schafer and Witthaut, "Pre-Processed Power Grid Frequency Time Series",
  Zenodo record 3744121, doi:10.5281/zenodo.3744121. File `Data_cleansed.zip`, member `Data_cleansed/Fingrid/2019.zip`
  (Zenodo checksum of the whole zip: md5:cfb08092728bdd0b24b1b71e590c4f3b).
- **Access date (UTC):** 2026-10-31
- **Licence:** CC BY 4.0 (Fingrid, Nordic; pre-processed version released by the record authors). The LICENSE.txt in the same folder of the zip says CC BY 4.0
  (https://creativecommons.org/licenses/by/4.0/). Attribution is required.
- **Not used:** every TransnetBW file (no open licence). None was downloaded.
- **How fetched:** `scripts/fetch_grid.py fetch`, with HTTP range requests, 3,148,547 bytes
  read from Zenodo. The slice is the first stretch that is complete and has no large event
  (every value within 0.25 Hz of 50 Hz, at most 100 NaN
  seconds a day). The rule is in the script.
- **File:** `data/raw/grid/fingrid_2019-01-01_5d.csv`
- **SHA-256:** `473d8b29605d92f0c63aaba8c8d2cc78e0842916aa8953574a5753c0536e9707`

## daphnia/data-and-code (extinctions.csv, timeseries.csv)

- **What:** weekly census of *Daphnia magna* populations and their extinction dates. Used as the
  split `real_tipping` (test only, surrogate labels). See `docs/DAPHNIA.md`.
- **Source:** Drake and Griffen (2010), "Data from: Early warning signals of extinction in
  deteriorating environments", Dryad, doi:10.5061/dryad.q3p64 (file `data-and-code.zip`,
  Dryad MD5 923e08e6bafae412c43250fef0752ac7). Paper: doi:10.1038/nature09389.
- **Access date:** 2026-10-31 (downloaded by hand from Dryad, which needs a login; unzipped into
  `data/raw/daphnia/`).
- **Licence:** CC0 1.0 (as given by the Dryad record).
- **Files used and SHA-256:**
  - `data/raw/daphnia/data-and-code/extinctions.csv`:
    `9446294b215b319544039c6d53818361aca60de902dd2e7060a552ab441ed7b9`
  - `data/raw/daphnia/data-and-code/timeseries.csv`:
    `e6bfc51d929ddcea7598749737313fcdf89fbcc8222c8780c2e8e918799037fd`
