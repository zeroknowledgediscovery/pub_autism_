#! /bin/bash
git lfs track *csv
git lfs track */*csv
git lfs track */*/*csv
git lfs track */*/*/*csv


git lfs track *zip
git lfs track */*zip
git lfs track */*/*zip
git lfs track */*/*/*zip
git lfs track */*/*/*/*zip
git lfs track */*/*/*/*/*zip


git add *csv -v
git commit -m csv
git push
git add SHAP/* -v
git commit -m SHAP
git push
git add * -v
git commit -m rest
git push
