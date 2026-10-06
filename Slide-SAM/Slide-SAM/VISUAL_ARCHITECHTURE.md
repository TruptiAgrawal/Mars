                    YOUR PROJECT
                         │
                         ▼
                    BTCV CT scan
                         │
                         ▼
                 ┌───────────────┐
                 │    ViT-B      │
                 │  SAM backbone │
                 └───────────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
           LoRA rank 8       LoRA-sub rank 1
                │                 │
                └────────┬────────┘
                         ▼
                  TRAINED MODEL
                         │
                model_iter_10000.pth
                         │
                         ▼
                  Load the numbers
                         │
                         ▼
                  Run the CT scan
                         │
                         ▼
                 Predicted liver
                     segmentation
                         │
                         ▼
                  Mask + CT image
                         │
                         ▼
                    PNG IMAGE