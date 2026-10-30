import argparse
import copy
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from accelerate.utils import set_seed
from src.utils import get_monai_transforms, build_dataset_file_list, collate_skip_none, collate_with_strings, instantiate_from_config, seed_everything, training_reproducibility_cudnn, seed_worker, init_optimizers, init_schedulers, find_latest_checkpoint, setup_output_dirs, load_checkpoint, delete_output_dir
from src.accelerator import find_executable_batch_size
from medlat import available_models, get_model
from src.datasets.medmnist import MergedMedMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as T

def create_dataloaders(config):

    train_transform = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.05, contrast=0.05),
        T.ToTensor(),
    ])

    eval_transform = T.Compose([
        T.ToTensor(),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
    ])

    print("Creating dataloaders...")
    train_loader = DataLoader(MergedMedMNIST(root=config.data_path, split="train", transform=train_transform, ds_names=config.ds_names), batch_size=config.bs, shuffle=True, num_workers=config.num_workers)
    print("Train loader created")
    eval_loader = DataLoader(MergedMedMNIST(root=config.data_path, split="val", transform=eval_transform, ds_names=config.ds_names), batch_size=config.bs, shuffle=False, num_workers=config.num_workers)
    print("Eval loader created")
    test_loader = DataLoader(MergedMedMNIST(root=config.data_path, split="test", transform=test_transform, ds_names=config.ds_names), batch_size=config.bs, shuffle=False, num_workers=config.num_workers)
    print("Test loader created")
    return train_loader, eval_loader, test_loader

def train(config: DictConfig):
    # Enable TensorFloat32 for better performance on Ampere+ GPUs
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
        print("Enabled TensorFloat32 for better performance")
    torch._dynamo.config.suppress_errors = True
    print(torch.utils.collect_env.main())



    accelerator = instantiate_from_config(config.accelerator)

    if accelerator.is_main_process and config.resume == "overwrite": delete_output_dir(config.base_output_dir)
    if accelerator.is_main_process: setup_output_dirs(config, config.base_output_dir)

    set_seed(config.seed + accelerator.process_index)
    seed_everything(config.seed + accelerator.process_index)
    training_reproducibility_cudnn()
    

    model = get_model(config.model, img_size=224)
    total_params = sum(p.numel() for p in model.parameters())
    if accelerator.is_main_process: print("Total parameters:", f"{total_params:,}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process: print("Trainable parameters:", f"{trainable_params:,}")
    config.total_params = total_params
    config.trainable_params = trainable_params


    ema_model = instantiate_from_config(config.ema, model=model) if 'ema' in config else None
    loss_fn = instantiate_from_config(config.loss) if hasattr(config, 'loss') else None
    
    latest_checkpoint = find_latest_checkpoint(config.base_output_dir) if config.ckpt_path is None else config.ckpt_path
    state = load_checkpoint(latest_checkpoint)

    
    @find_executable_batch_size(starting_batch_size=config.bs)
    def train_with_batch_size(batch_size):
        nonlocal state, model, ema_model, loss_fn
        
        print(f"Using adapted batch size: {batch_size}")

        config.bs = batch_size
        train_loader, eval_loader, test_loader = create_dataloaders(config)

        optimizer = init_optimizers(config, model, loss_fn)
        scheduler = init_schedulers(config, optimizer, num_processes=accelerator.num_processes, epoch_to_steps=len(train_loader))
                

        if accelerator.is_main_process and not config.debug:  # Only log if not in debug mode
            wandb_init_kwargs = {
                "name": config.experiment_name,
                "resume": "allow",  # Allow resuming runs
                "allow_val_change": True  # Allow changing values for existing steps
            }
            
            # If we have a wandb run ID in the state, use it to resume
            if state and 'wandb_run_id' in state:
                print(f"Resuming run with ID: {state['wandb_run_id']}")
                wandb_init_kwargs["id"] = state['wandb_run_id']
            
            # Convert the config to a dictionary and log it to wandb
            config_dict = OmegaConf.to_container(config, resolve=True)
            wandb_init_kwargs["config"] = config_dict
            
            accelerator.init_trackers(
                project_name=config.project_name,
                config=config_dict,
                init_kwargs={"wandb": wandb_init_kwargs}
            )

            wandb.watch(model, log="all", log_freq=100)  
               
        trainer_params = {
            'model': model,
            'ema_model': ema_model,
            'train_loader': train_loader,
            'eval_loader': eval_loader,
            'optimizers': optimizer,
            'schedulers': scheduler,
            'output_dir': str(config.base_output_dir),
            'loss_fn': loss_fn,
            'accelerator': accelerator,
            'diffusion': instantiate_from_config(config.diffusion) if 'diffusion' in config else None,
            'early_stopper': instantiate_from_config(config.earlystop) if 'earlystop' in config else None,
            'evaluator': instantiate_from_config(config.evaluator) if 'evaluator' in config else None,
            **config.trainer.get('params', {})
        }

        trainer = instantiate_from_config(config.trainer, **trainer_params)
        if state:
            trainer.load_state_dict(copy.deepcopy(state))
        # trainer.model = torch.compile(trainer.model, mode="default")
        print(trainer)
        trainer.train()
        ## dataloader stress test 
        # for batch in tqdm(train_loader):
        #     pass

    train_with_batch_size()

def resolve_ckpt(ckpt, base_output_dir):
    """`--ckpt 3` -> <run>/train/models/ckpt.3.pt; any other value is a path."""
    if ckpt is None:
        return None
    return f"{base_output_dir}/train/models/ckpt.{ckpt}.pt" if str(ckpt).isdigit() else ckpt


def main():
    parser = argparse.ArgumentParser(description='Train a tokenizer (autoencoder + quantizer) on the merged MedMNIST+ 2D datasets.')
    parser.add_argument("--model", type=str, required=True,
                        help="Tokenizer name in the medlat registry, e.g. discrete.rqvae.f8_d4_e16384.")
    parser.add_argument("--config", type=str, default='configs/tokenizer.yaml', help="Path to the YAML config.")
    parser.add_argument("--resume", type=str, default="auto",
                        help="'auto' resumes from the latest checkpoint; 'overwrite' starts fresh.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint to resume from: an index (e.g. 3) or a path.")
    parser.add_argument("--bs", type=int, default=None, help="Per-GPU batch size (default: from config).")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_wandb", "--debug", dest="debug", action="store_true",
                        help="Disable Weights & Biases logging.")
    parser.add_argument("--exp_name", type=str, default=None, help="Run name (default: derived from --model).")

    # Unknown args are OmegaConf dot-list overrides, e.g. optimizer.gen.lr=1e-4
    args, unknown = parser.parse_known_args()
    unknown = [arg.lstrip('-') if arg.startswith('-') else arg for arg in unknown]

    config = OmegaConf.load(args.config)
    args_conf = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None})
    config = OmegaConf.merge(config, args_conf, OmegaConf.from_cli(unknown))

    exp_name = args.exp_name or (lambda model: model)(args.model)
    config.experiment_name = exp_name
    config.base_output_dir = f"{config.base_dir}/outputs/{config.project_name}/{exp_name}"
    config.ckpt_path = resolve_ckpt(args.ckpt, config.base_output_dir)
    print(OmegaConf.to_yaml(config))
    print("Base output directory:", config.base_output_dir)

    train(config)


if __name__ == '__main__':
    main()
