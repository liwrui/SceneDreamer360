import os
import torch
import wandb
import argparse
import json
# from models import *
# from dataset import *
from PanFusion import *
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.trainer import Trainer
from datetime import timedelta

from PIL import Image
from Enhance_img import Text2360PanoramaImagePipeline

from multi_view_img import *
from PanoSpaceDreamer.utils.trajectory import *

# ld
import re
from PanoSpaceDreamer.luciddreamer import LucidDreamer



class ConfigLoader:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            return json.load(file)

    def get(self, key, default=None):
        return self.config.get(key, default)



def cli_main():
    config_file = 'config.json'
    config_loader = ConfigLoader(config_file)

    text = config_loader.get('text', './data/Matterport3D/mp3d_skybox/e9zR4mvMWw7/blip3_stitched/test.txt')
    neg_text = config_loader.get('neg_text', '')
    campath_gen = config_loader.get('campath_gen', 'fullscan')
    campath_render = config_loader.get('campath_render', '1440')
    model_name = config_loader.get('model_name', None)
    seed = config_loader.get('seed', 1)
    diff_steps = config_loader.get('diff_steps', 50)
    save_dir = config_loader.get('save_dir', '')
    image_size = int(config_loader.get('image_size', '768'))


    # Pano_Image Generation
    if 'SLURM_NTASKS' in os.environ:
        del os.environ["SLURM_NTASKS"]
    if 'SLURM_JOB_NAME' in os.environ:
        del os.environ["SLURM_JOB_NAME"]

    torch.set_float32_matmul_precision('medium')

    wandb_id = os.environ.get('WANDB_RUN_ID', wandb.util.generate_id())
    exp_dir = os.path.join('logs', wandb_id)
    os.makedirs(exp_dir, exist_ok=True)
    wandb_logger = lazy_instance(
        WandbLogger,
        project='panfusion',
        id=wandb_id,
        save_dir=exp_dir
        )

    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        save_last=True,
        train_time_interval=timedelta(minutes=10),
        )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    class MyLightningCLI(LightningCLI):
        def before_instantiate_classes(self):
            # set result_dir, data and pano_height for evaluation
            if self.config.get('test', {}).get('model', {}).get('class_path') == 'models.EvalPanoGen':
                if self.config.test.data.init_args.result_dir is None:
                    result_dir = os.path.join(exp_dir, 'test')
                    self.config.test.data.init_args.result_dir = result_dir
                self.config.test.model.init_args.data = self.config.test.data.class_path.split('.')[-1]
                self.config.test.model.init_args.pano_height = self.config.test.data.init_args.pano_height
                self.config.test.data.init_args.batch_size = 1

        def add_arguments_to_parser(self, parser):
            parser.link_arguments("model.init_args.cam_sampler", "data.init_args.cam_sampler")

    cli = MyLightningCLI(
        trainer_class=Trainer,
        save_config_kwargs={'overwrite': True},
        parser_kwargs={'parser_mode': 'omegaconf', 'default_env': True},
        seed_everything_default=os.environ.get("LOCAL_RANK", 0),
        trainer_defaults={
            'strategy': 'ddp',
            'log_every_n_steps': 10,
            'num_sanity_val_steps': 0,
            'limit_val_batches': 4,
            'benchmark': True,
            'max_epochs': 10,
            'precision': 32,
            'callbacks': [checkpoint_callback, lr_monitor],
            'logger': wandb_logger
        }
        )
    
    # cli.run()
    # Pano_img and propt save_dir
    text_dir_name = text.split('/')[-1].split('.')[0]
    result_dir = f'../logs/4142dlo4/predict/e9zR4mvMWw7_{text_dir_name}'
    

    # Enhance Pano_Image
    # prompt_path = result_dir + '/pano_image/prompt.txt'
    # with open(prompt_path) as f:
    #     prompt = f.read()

    
    with open(text, 'r', encoding='utf-8') as file:
        prompt = file.read().strip()

    
    print(f'prompt:{prompt}')


    pano_img_path = result_dir + '/pano.jpg'
    pano_img = Image.open(pano_img_path)
    input = {'prompt': prompt, 'pano_image': pano_img, 'upscale': True}

    model_id = 'Enhance_img/models'
    txt2panoimg = Text2360PanoramaImagePipeline(model_id)
    # output = txt2panoimg(input)
    output = pano_img
    output.save(result_dir + '/pano_enhance.jpg')

    
    
    # multi_view image
    render_poses = get_pcdGenPoses(campath_gen)

    # pano_img path
    pano_img_path = result_dir + '/pano_enhance.jpg'

    # multi_view image size
    image_size = image_size

    
    multi_view_image_list_dir = result_dir + '/multi_view_image_list'
    if not os.path.exists(multi_view_image_list_dir):
        os.makedirs(multi_view_image_list_dir,exist_ok=True)

    rgb_cond_list = []
    for i in range(len(render_poses)):
        R = render_poses[i, :3, :3]
        T = render_poses[i, :3, 3:4]

        R = R.T
        T = -T
        
        output_size = (image_size, image_size)
        perspective_image = get_perspective_image(pano_img_path, R, T, output_size=output_size)
        cv2.imwrite(multi_view_image_list_dir + f'/{i}.jpg',perspective_image)

        perspective_image_pil = Image.fromarray(cv2.cvtColor(perspective_image, cv2.COLOR_BGR2RGB))
        rgb_cond_list.append(perspective_image_pil)

    
    if text.endswith('.txt'):
        with open(text, 'r') as f:
            txt_cond = f.readline()
    else:
        txt_cond = text

    if neg_text.endswith('.txt'):
        with open(neg_text, 'r') as f:
            neg_txt_cond = f.readline()
    else:
        neg_txt_cond = neg_text

    # Make default save directory if blank
    if save_dir == '':
        save_dir = result_dir + f'/outputs/{campath_gen}_{seed}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if model_name is not None and model_name.endswith('safetensors'):
        print('Your model is saved in safetensor form. Converting to HF models...')
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt

        pipe = download_from_original_stable_diffusion_ckpt(
            checkpoint_path_or_dict=model_name,
            from_safetensors=True,
            device='cuda',
            )
        pipe.save_pretrained('stablediffusion/', safe_serialization=False)
        model_name = f'stablediffusion/{model_name}'

    depth_save_dir = result_dir + '/depth_multi_view_image'
    if not os.path.exists(depth_save_dir):
        os.makedirs(depth_save_dir, exist_ok=True)
    
    ld = LucidDreamer(image_size=image_size,for_gradio=False, save_dir=save_dir,depth_save_dir=depth_save_dir)
    ld.create(rgb_cond_list, txt_cond, neg_txt_cond, campath_gen, seed, diff_steps, model_name=model_name)
    ld.render_video(campath_render)


if __name__ == '__main__':
    cli_main()
