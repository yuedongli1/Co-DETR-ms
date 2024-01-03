import mindspore as ms

paramdict_ms = ms.load_checkpoint('./co_dino_swin_large_all_heads.ckpt')
paramdict_torch = ms.load_checkpoint('./co_dino_swin_large_all_heads_torch.ckpt')
diff = 0
for k, v in paramdict_torch.items():
    if k not in paramdict_ms:
        diff += 1
        print(diff, k)
