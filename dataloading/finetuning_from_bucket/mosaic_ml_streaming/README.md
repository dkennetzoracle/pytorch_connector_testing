# MosaicML FineTuning

This needs to be run on each machine prior to launch to properly configure the setup:
```bash
accelerate config --config_file "fsdp_config.yaml"

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------In which compute environment are you running?
This machine                                                                                               

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?                                                                                                                                                                                
multi-GPU                                                                                                                                                   

How many different machines will you use (use more than 1 for multi-node training)? [1]: 2                                                                                             

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------What is the rank of this machine?                                                                                                                                                                          
0                                                                                                                                                           

What is the IP address of the machine that will host the main process? 10.140.19.72                                                                                      

What is the port you will use to communicate with the main process? 12355                                                                            

Are all the machines on the same local network? Answer `no` if nodes are on the cloud and/or on different network hosts [YES/no]:                                                                                   

Should distributed operations be checked while running for errors? This can avoid timeout issues but will be slower. [yes/NO]:                                                                                 

Do you wish to optimize your script with torch dynamo?[yes/NO]:

Do you want to use DeepSpeed? [yes/NO]: 

Do you want to use FullyShardedDataParallel? [yes/NO]: yes

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------What should be your sharding strategy?
FULL_SHARD                                                                                  

Do you want to offload parameters and gradients to CPU? [yes/NO]:                                                                                                                                                   

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------What should be your auto wrap policy?                                                                                                                                                                               
TRANSFORMER_BASED_WRAP                                                                                                                                                 

Do you want to use the models `_no_split_modules` to wrap. Only applicable for 🤗 Transformers [yes/NO]: yes                                                                                 

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------What should be your FSDPs backward prefetch policy?                                                                                                                                                                
BACKWARD_PRE                                                                                  

--------------------------------------------------------------------------------------------------------------------------------------------------------------------What should be your FSDPs state dict type?                                                                                                                                                                         
SHARDED_STATE_DICT      

Do you want to enable FSDPs forward prefetch policy? [yes/NO]: yes                                                                                                                                                 

Do you want to enable FSDPs `use_orig_params` feature? [YES/no]: yes                                                                                             

Do you want to enable CPU RAM efficient model loading? Only applicable for 🤗 Transformers models. [YES/no]:            

Do you want to enable FSDP activation checkpointing? [yes/NO]: 

How many GPU(s) should be used for distributed training? [1]:8

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------Do you wish to use mixed precision?
bf16                                                                                                                                                     

accelerate configuration saved at fsdp_config.yaml 
```
Results in config:
```yaml
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_process_ip: 10.140.19.72
main_process_port: 12355
main_training_function: main
mixed_precision: bf16
num_machines: 2
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```