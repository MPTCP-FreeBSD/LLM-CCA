# Preface
The code for this project is based on the [NetLLM](https://github.com/duowuyms/NetLLM) repository. Thanks to the NetLLM team for providing the open source code!

# Code Structure
- `artifacts`: This directory stores some artifacts, e.g., result files.
   - `exp_pool`: This directory stores the experience pool files, which will be used for LLM adaptation.
   - `results`: This directory stores the result files.

- `data`: This directory stores datasets and pre-trained model checkpoints of baselines.
   - `traces`: This directory stores the bandwidth trace datasets.
   - `videos`: This directory stores the video specifications.
   - `ft_plms`: This directory stores the fine-tuned (adapted) LLMs.
   - `all_models`: This directory stores the model checkpoints of baselines.

- `baseline_specical`: This directory stores the codes for runing baselines. Most of the codes are from the Genet's repository.
- `plm_special`: This directory stores the codes for running NetLLM.
   - `data`: This directory stores the codes related to the training datasets for LLM adaptation.
      - `exp_pool.py`: Implements the experience pool for collecting trajectories.
      - `dataset.py`: Implements a dataset class that wraps the experience pool.
    - `models`: This directory stores the codes related to NetLLM.
      - `state_encoder.py`: Implements the feature encoder for encoding states.
      - `gpt2.py, llama.py, opt.py, mistral.py, t5.py`: Customized LLMs.
      - `low_rank.py`: Implements the low rank matrices.
      - `rl_policy.py`: Implements the Transformer-based offline RL policy.
    - `utils`: This directory stores some utilization codes.
      - `plm_utils.py`: Some codes for loading LLMs.
      - `utils.py`: Some codes for data processing.
    - `trainer.py`: Some codes for training (adapting) LLMs. 
    - `evaluate.py`: Some codes for evaluting the performance of adapted-LLMs.
    - `test.py`: Some codes for testing the performance of adapted LLMs.

- `PKL`: This directory is used to store experience pool files, including the data generated while running different algorithms under the L4S architecture.
   The data was collected under conditions where the BBR, Prague, and CUBIC congestion control algorithms were running simultaneously.
   - `1000bbr_exp_pool.pkl`: PKL file generated from running 1000 instances of the BBR algorithm under the L4S architecture.
   - `1000cubic_exp_pool.pkl`: PKL file generated from running 1000 instances of the CUBIC algorithm under the L4S architecture.
   - `1000prague_exp_pool.pkl`: PKL file generated from running 1000 instances of the Prague algorithm under the L4S architecture.
   - `3000bbr_exp_pool.pkl`: PKL file generated from running 3000 instances of the BBR algorithm under the L4S architecture.
   - `3000cubic_exp_pool.pkl`: PKL file generated from running 3000 instances of the CUBIC algorithm under the L4S architecture.
   - `3000prague_exp_pool.pkl`: PKL file generated from running 3000 instances of the Prague algorithm under the L4S architecture.
   - `5000bbr_exp_pool.pkl`: PKL file generated from running 5000 instances of the BBR algorithm under the L4S architecture.
   - `5000cubic_exp_pool.pkl`: PKL file generated from running 5000 instances of the CUBIC algorithm under the L4S architecture.
   - `5000prague_exp_pool.pkl`: PKL file generated from running 5000 instances of the Prague algorithm under the L4S architecture.

- `my exp_pool_code.py`: Implements the generation of experience pool (i.e., training dataset for LLM).
- `run_baseline.py`: The main file for running baselines. 
- `read_pkl.py`: Converts pkl files to txt format for easier verification of the file contents.

# Environment Setup
## Environment for NetLLM
1. Create a conda environment for NetLLM:

   `conda create -n abr_netllm python>=3.8.10`

2. Activating the Conda environment
   ```
   conda activate abr_netllm
   ```

3. Then install the following depdendencies one by one:

   ```
   python==3.8.10
   torch==2.1.0
   numpy==1.24.4
   munch==4.0.0
   openprompt==1.0.1
   transformers==4.34.1
   peft==0.6.2
   ```

4. Alternatively, you can install everything at once using the following command:

```
python -m pip install --upgrade pip && pip install openprompt==1.0.1 && pip install numpy==1.24.4 && pip install peft==0.6.2 && pip install transformers==4.34.1 && pip install --upgrade huggingface_hub && pip install scikit-learn && pip install munch
```

# Usage
## Usage of NetLLM
To run NetLLM, first we need to download some LLMs. For example, if you want to use Llama2-7b as the foundation model, please download Llama2-7b in the directory: `../downloaded_plms/llama2/base`. In the following, we will use the Llama2-7b as the example to illustrate the usage of NetLLM.

**Finetune LLM**

If you want to finetune LLM, please run the following command.:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --device-out cuda:1 --lr 0.0001 --warmup-steps 2000 --num-epochs 20 --eval-per-epoch 2 --exp-pool-path ./PKL/1000bbr_exp_pool.pkl 
```
This command will finetune Llama2 on the default experience pool we provided at `PKL/1000bbr_exp_pool.pkl`. You can also switch to your desired PKL file based on the data you want to use
If you want to use your own experience pool, first use the `my_exp_pool_code.py` to generate a new experience pool.
```sh
python my_exp_pool_code.py
```
Next, specify the path to your own experience pool with argument `--exp-pool-path` and run the following command:
```sh
python run_plm.py --adapt --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2--exp-pool-path your_exp_pool_path
```

**Test LLM**

If you want to test the performance of the finetuned LLM, please run the following command:
```sh
python run_plm.py --test --grad-accum-steps 32 --plm-type llama --plm-size base --rank 128 --device cuda:0 --lr 0.0001 --warmup-steps 2000 --num-epochs 80 --eval-per-epoch 2
```
You can also specify the path to the finetuned LLM with argument `--model-dir`:
```sh
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir you_finetune_llm_dir
```

We offer the model checkpoint of the finetuned Llama2-7b here: https://drive.google.com/file/d/17UyXJ9rGc0wKUkAhQ4wMrYDEbRPRjil0/view. If you want to try our model, please download the model checkpoint and store it in `data/ft_plms/try_llama2_7b`, and run the following command:
```sh
python run_plm.py --test --plm-type llama --plm-size base --rank 128 --device cuda:0 --model-dir  data/ft_plms/try_llama2_7b
```



