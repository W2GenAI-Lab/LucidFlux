result_dir=ouput_images_folder
input_folder=input_images_folder
checkpoint_path=path/to/lucidflux.pth
swin_ir_ckpt=path/to/swinir.ckpt
siglip_ckpt=path/to/siglip.ckpt

mkdir -p ${result_dir}
echo "Processing checkpoint..."
python inference.py \
  --checkpoint ${checkpoint_path} \
  --swinir_pretrained ${swin_ir_ckpt} \
  --control_image ${input_folder} \
  --siglip_ckpt ${siglip_ckpt} \
  --prompt "restore this image into high-quality, clean, high-resolution result" \
  --output_dir ${result_dir}/ \
  --width 1024 --height 1024 --num_steps 50 \