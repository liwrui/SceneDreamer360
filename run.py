import os
import json
import subprocess

def main():
    prompt_file = './data/prompt.txt'
    test_file_root = './data/Matterport3D/mp3d_skybox/e9zR4mvMWw7/blip3_stitched/'
    test_file_name = 'test.txt'
    config_file = './config.json'
    log_file = './logs/log.txt'
    conda_env = os.getenv('CONDA_DEFAULT_ENV')
    print(conda_env)

    with open(prompt_file, 'r', encoding='utf-8') as file:
        prompts = file.readlines()

    for i, prompt in enumerate(prompts):
        new_test_file_name = f'{i+11}.txt'
        os.rename(test_file_root + test_file_name, test_file_root + new_test_file_name)
        test_file_name = new_test_file_name
        new_test_file_path = test_file_root + new_test_file_name

        with open(new_test_file_path, 'w', encoding='utf-8') as file:
            file.write(prompt)

        with open(config_file, 'r', encoding='utf-8') as file:
            config_data = json.load(file)

        config_data['text'] = new_test_file_path
        
        with open(config_file, 'w', encoding='utf-8') as file:
            json.dump(config_data, file, ensure_ascii=False, indent=4)
        
        command = 'WANDB_MODE=offline WANDB_RUN_ID=4142dlo4 python main.py predict --data=Matterport3D --model=PanFusion --ckpt_path=last'
        result = subprocess.run(command, shell=True)
        
        with open(log_file, 'a', encoding='utf-8') as log:
            log.write(f'Executed command for prompt {i+1}: {prompt.strip()}\n')
            log.write(f'Result: {result.returncode}\n\n')

if __name__ == "__main__":
    main()
