# natix_classifer

## Scripts

### train
Single-GPU, two models, one train dir one val dir:
```bash
chmod +x run_train.sh
./scripts//run_train.sh "convnext_small,resnet18" "/data/train" "/data/val" "checkpoints" "tb_logs" "0"
```

Multi-train dirs, multi-val dirs:
```bash
./scripts/run_train.sh "convnext_small,efficientnet_b3" "/workspace/hf_extracted_images/natix-network-org_roadwork:/workspace/synthetic/i2i:/workspace/synthetic/t2i" 0.1 "checkpoints" "tb_logs" "0,1"
or
vit_base+deit_small+swin_small+swin_v2_small+
python train.py --models=efficientnetv2_s+resnet18+resnet50 --train-dirs="/workspace/hf_extracted_images/train"+"/workspace/hf_extracted_images/test"+"/workspace/synthetic/i2i"+"/workspace/synthetic/t2i" --val-split=0.1 --tb-logdir=tb_logs --pretrained --augment | tee efficientnet_v2_resnet18_50.log
```

```bash
tensorboard --logdir tb_logs
```

### test
```bash
chmod +x run_test.sh
./scripts//run_test.sh "checkpoints/convnext_small_best.pth checkpoints/resnet18_best.pth" convnext_small "/data/test" "tb_test" "0"
```