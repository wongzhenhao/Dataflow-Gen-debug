import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

from src.config import init_config
from src.core.Wrapper import ImageCaptionerWrapper, VideoCaptionerWrapper, ImageGeneratorWrapper, VideoGeneratorWrapper, TextGeneratorWrapper

def Image_Captioner(cfg):
    captioner = ImageCaptionerWrapper(
                    meta_path=cfg.meta_path, 
                    image_folder=cfg.meta_folder, 
                    save_folder=cfg.save_folder, 
                    save_per_batch=cfg.save_per_batch,
                )
    for captioner_name, model_config in cfg.ImageCaptioner.items():
        captioner.generate_for_one_model(
                            model_name=captioner_name, 
                            batch_size=model_config['batch_size'],
                            model_config=model_config
                    )

def Image_Generator(cfg):
    generator = ImageGeneratorWrapper(
                    meta_path=cfg.meta_path, 
                    save_folder=cfg.save_folder, 
    )
    for generator_name, model_config in cfg.ImageGenerator.items():
        generator.generate_for_one_model(
                            model_name=generator_name, 
                            batch_size=model_config['batch_size'],
                            model_config=model_config,
                    )

def Text_Generator(cfg):
    generator = TextGeneratorWrapper(
                    meta_path=cfg.meta_path, 
                    save_folder=cfg.save_folder, 
                    save_file=cfg.save_file
    )
    for generator_name, model_config in cfg.TextGenerator.items():
        generator.generate_for_one_model(
                            model_name=generator_name, 
                            batch_size=model_config['batch_size'],
                            model_config=model_config,
                    )

def Video_Captioner(cfg):
    captioner = VideoCaptionerWrapper(
                    meta_path=cfg.meta_path, 
                    video_folder=cfg.meta_folder, 
                    save_folder=cfg.save_folder, 
                    save_per_batch=cfg.save_per_batch,
                )
    for captioner_name, model_config in cfg.VideoCaptioner.items():
        captioner.generate_for_one_model(
                            model_name=captioner_name, 
                            batch_size=model_config['batch_size'],
                            model_config=model_config
                    )

def Video_Generator(cfg):
    generator = VideoGeneratorWrapper(
                    meta_path=cfg.meta_path, 
                    save_folder=cfg.save_folder, 
    )
    for generator_name, model_config in cfg.VideoGenerator.items():
        generator.generate_for_one_model(
                            model_name=generator_name, 
                            batch_size=model_config['batch_size'],
                            model_config=model_config
                    )
                            
def main():
    cfg = init_config()
    
    if cfg.ImageCaptioner:
        Image_Captioner(cfg)
    if cfg.ImageGenerator:
        Image_Generator(cfg)
    if cfg.TextGenerator:
        Text_Generator(cfg)
    if cfg.VideoCaptioner:
        Video_Captioner(cfg)
    if cfg.VideoGenerator:
        Video_Generator(cfg)

if __name__ == '__main__':
    main()