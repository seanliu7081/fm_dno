#!/usr/bin/env python3
"""Full fine-tune QwenVL and Heading Gaussian DiT with six torchrun ranks."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'oat/config/starvla_heading_gaussian.yaml')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--stop-after-steps', type=int, help='Save and pause at this optimizer step without changing immutable training settings')
    parser.add_argument('--gradient-accumulation-steps', type=int, help='Profile override; production default is ten')
    parser.add_argument('--micro-batch-size', type=int, help='Per-GPU microbatch override; keep six times microbatch times accumulation equal to the intended global batch')
    parser.add_argument('--validation-samples', type=int)
    parser.add_argument('--validation-interval', type=int)
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--log-interval', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--skip-resume-save', action='store_true',
                        help='Profile only: export weights but omit the full optimizer checkpoint')
    parser.add_argument('--print-config', action='store_true')
    args = parser.parse_args()
    from omegaconf import OmegaConf
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    for argument, section, key in (
        (args.max_steps, 'training', 'max_steps'), (args.num_workers, 'training', 'num_workers'),
        (args.gradient_accumulation_steps, 'training', 'gradient_accumulation_steps'),
        (args.micro_batch_size, 'training', 'micro_batch_size'),
        (args.validation_samples, 'validation', 'num_samples'),
        (args.validation_interval, 'validation', 'interval'),
        (args.checkpoint_interval, 'training', 'checkpoint_interval'),
        (args.log_interval, 'training', 'log_interval')):
        if argument is not None:
            config[section][key] = argument
    if args.output_dir is not None:
        config['output_dir'] = str(args.output_dir.resolve())
    if args.manifest is not None:
        config['dataset']['manifest'] = str(args.manifest.resolve())
    if args.seed is not None:
        config['seed'] = args.seed
    from oat.starvla_heading.training import run_training, validate_configuration
    from oat.starvla_heading.data import load_manifest
    validate_configuration(config, load_manifest(config['dataset']['manifest']))
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    run_training(config, resume=args.resume, skip_resume_save=args.skip_resume_save,
                 stop_after_steps=args.stop_after_steps)


if __name__ == '__main__':
    main()
