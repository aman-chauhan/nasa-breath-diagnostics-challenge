# nasa-breath-diagnostics-challenge
Enhance NASA's E-Nose for Accurate Medical Diagnostics

## Steps to run

```bash
pip install -r requirements.txt
sh setup.sh
# extract the zip file inside raw_data folder
python -m scripts.collate
python -m scripts.clean
```