PYTORCH_ENABLE_MPS_FALLBACK=1
# ---------- oscc_bin ----------
sed -i '' 's/project: dys_bin/project: oscc_bin/' params.yaml
# mobilenetv2
sed -i '' 's/name: vgg16/name: mobilenetv2/' params.yaml
python3 main.py

# densenet121
sed -i '' 's/name: mobilenetv2/name: densenet121/' params.yaml
python3 main.py

# resnet50
sed -i '' 's/name: densenet121/name: resnet50/' params.yaml
python3 main.py

# vgg16
sed -i '' 's/name: resnet50/name: vgg16/' params.yaml
python3 main.py

# ---------- dys_bin ----------
sed -i '' 's/project: oscc_bin/project: dys_bin/' params.yaml

# vgg16
python3 main.py

# resnet50
sed -i '' 's/name: vgg16/name: resnet50/' params.yaml
python3 main.py

# densenet121
sed -i '' 's/name: resnet50/name: densenet121/' params.yaml
python3 main.py

# mobilenetv2
sed -i '' 's/name: densenet121/name: mobilenetv2/' params.yaml
python3 main.py