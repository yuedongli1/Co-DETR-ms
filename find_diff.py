import mindspore as ms

paramdict_ms = ms.load_checkpoint('./co_dino_5scale_swin_large_16e_o365tococo.ckpt')
paramdict_torch = ms.load_checkpoint('./co_dino_5scale_swin_large_torch.ckpt')
diff = 0
for k, v in paramdict_torch.items():
    if k not in paramdict_ms:
        diff += 1
        print(diff, k)
