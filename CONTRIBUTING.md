# Contributing

## Pushing code to master

Stay updated with the `master` branch to get the latest version of all the changes.

`master` is not protected, so please be careful.  **Always request review for PRs to `master` from an experienced person on the team before merging.**

Details for how to contribute to each of the more important parts of the code are outlined below.

### Workflows
Each analysis has its own ntuplemaker and utilities. For the ntuplemakers, **unit tests are provided** in `additional_tools/unit_tests/test_ntuplemaker.py`. Please run these when modifying anything under `workflows/`, even if you have only modified utilities for your own analyses, as other scripts might call them too.

### Histmakers
For the histmaking scripts, no unit tests are available. The histmaker should work as a general tool for all analyses.

### Plotting
Plotting notebooks or scripts can be modified freely for each analysis. The `plot.ipynb` should be left as an example of what can be done.

## Issues

If you find bugs, want a new feature, have a new idea, etc., please open a Github issue in this repo.
