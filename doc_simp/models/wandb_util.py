import os


def existing_checkpoints(args):
    """Check if there are existing checkpoints matching configuration."""
    base = "" if args.save_dir is None else args.save_dir
    proj_dir = os.path.join(base, args.project, args.wandb_id, "checkpoints")
    if os.path.isdir(proj_dir):
        for file in os.listdir(proj_dir):
            if file.endswith(".ckpt"):
                return True
    return False