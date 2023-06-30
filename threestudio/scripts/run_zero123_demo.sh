python launch.py --config configs/zero123.yaml --train --gpu 7 system.loggers.wandb.enable=false system.loggers.wandb.project="voletiv-anya-new" system.loggers.wandb.name="dragon2" data.image_path=./load/images/dragon2_rgba.png system.freq.guidance_eval=0 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" tag='${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}_XL'

python threestudio/scripts/make_training_vid.py --exp /admin/home-vikram/git/threestudio/outputs/zero123/64_dragon2_rgba.png@20230628-152734 --frames_per_vid 30 --fps 20 --max_iters 200

# Phase 2
python launch.py --config configs/zero123_512.yaml --train --gpu 5 system.loggers.wandb.enable=true system.loggers.wandb.project="voletiv-zero123XL-demo" system.loggers.wandb.name="robot_512_drel_n_XL_SAMEgeom" data.image_path=./load/images/robot_rgba.png system.freq.guidance_eval=0 system.guidance.pretrained_model_name_or_path="./load/zero123/XL_20230604.ckpt" tag='${data.random_camera.height}_${rmspace:${basename:${data.image_path}},_}_XL_SAMEgeom' system.geometry_convert_from="/admin/home-vikram/git/threestudio/outputs/zero123/[64, 128]_robot_rgba.png_OLD@20230630-052314/ckpts/last.ckpt"
