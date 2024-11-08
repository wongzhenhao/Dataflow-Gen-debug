import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
    datefmt='%m-%d %H:%M:%S'
)

from src.config import init_generation_config
from src.core.GeneratorWrapper import GeneratorWrapper
def main():
    cfg = init_generation_config()
    generator = GeneratorWrapper(
                    meta_prompt_path=cfg.meta_prompt_path, 
                    save_folder=cfg.save_folder, 
    )
    # generate for each model
    for generator_name, model_config in cfg.generators.items():
        generator.generate_for_one_model(
                            model_name=generator_name, 
                            model_config=model_config)
        
    

if __name__ == '__main__':
    main()
