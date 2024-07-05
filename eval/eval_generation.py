import torch_fidelity
# /data1/home/yuanzhouhang/Result/StyleGAN2-ada/ODIR/generation/AMD
real_image_folder = 'INSERT_ORIGINAL_IMAGES_PATH'
generated_image_folder = 'INSERT_GENERATED_IMAGES_PATH'

print("Evaluating FID, IS...")
metrics_dict = torch_fidelity.calculate_metrics(
    input1=real_image_folder,
    input2=generated_image_folder,
    cuda=True,
    isc=True,
    fid=True,
    kid=False,
    prc=False,
    verbose=False,
)
print(metrics_dict.keys())
fid = metrics_dict['frechet_inception_distance']
inception_score = metrics_dict['inception_score_mean']
# ppl = metrics_dict['perceptual_path_length']
print("FID: {}, IS: {}".format(fid, inception_score))
