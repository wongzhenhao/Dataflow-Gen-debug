import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

from src.config import init_config
from src.core.CaptionerWarpper import CaptionerWarpper
def main():
    cfg = init_config()
    
    captioner = CaptionerWarpper(
                    meta_image_path=cfg.meta_image_path, 
                    image_folder=cfg.image_folder, 
                    save_folder=cfg.save_folder, 
                    save_per_batch=cfg.save_per_batch,
                )
    for captioner_name, model_config in cfg.captioners.items():
        captioner.generate_for_one_model(
                            model_name=captioner_name, 
                            repeat_time=model_config['repeat_time'], 
                            batch_size=model_config['batch_size'],
                            model_config=model_config
                    )

if __name__ == '__main__':
    main()