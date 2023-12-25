import mindspore as ms

param_dict = ms.load_checkpoint('./co_dino_5scale_swin_large_16e_o365tococo.ckpt')
print('done')
