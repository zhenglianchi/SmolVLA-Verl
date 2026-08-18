# -*- coding: utf-8 -*-
"""Register the smolvla architecture in verl-vla's builder + model config."""
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
builder = repo / "src/verl_vla/models/builder.py"
modelcfg = repo / "src/verl_vla/workers/config/model.py"


def patch(path: Path, old: str, new: str) -> bool:
    s = path.read_text(encoding="utf-8")
    if new in s:
        print("SKIP (already patched)", path)
        return False
    if old in s:
        path.write_text(s.replace(old, new, 1), encoding="utf-8")
        print("PATCHED", path)
        return True
    print("SKIP (anchor missing)", path)
    return False


builder_branch = '''    if architecture == "smolvla":
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.pretrained import SAFETENSORS_SINGLE_FILE

        from .smolvla import SmolVLAConfig, SmolVLAPolicy, SmolVLATrainableModel, load_smolvla_processors

        if overrides:
            raise ValueError("SmolVLA architecture is checkpoint-owned; model.override_config must be empty")
        model_path = Path(path)
        weights_path = model_path / SAFETENSORS_SINGLE_FILE
        initialization_path = model_path / "initialization.json"
        if weights_path.is_file():
            policy = SmolVLAPolicy.from_pretrained(path)
        else:
            if not initialization_path.is_file():
                raise FileNotFoundError(
                    f"Native SmolVLA weights are missing at {weights_path}. Config-only initialization "
                    f"requires an explicit {initialization_path.name} sidecar."
                )
            with initialization_path.open(encoding="utf-8") as file:
                initialization = json.load(file)
            if initialization != {"type": "smolvla_config"}:
                raise ValueError(f"Unsupported SmolVLA initialization metadata in {initialization_path}")
            config = PreTrainedConfig.from_pretrained(path)
            if not isinstance(config, SmolVLAConfig):
                raise TypeError(f"Expected a native SmolVLA config at {path}, got {type(config).__name__}")
            policy = SmolVLAPolicy(config)
        adapter_config = dict(model_config.adapter)
        processor_dataset_root = adapter_config.pop("processor_dataset_root", None)
        preprocessor, postprocessor = load_smolvla_processors(
            policy.config,
            model_path,
            dataset_root=processor_dataset_root,
        )
        policy.to(dtype=torch_dtype)
        return SmolVLATrainableModel(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            adapter_config=adapter_config,
        )

    if architecture == "gaussian_actor":'''
patch(builder, '    if architecture == "gaussian_actor":', builder_branch)

patch(
    modelcfg,
    '        elif policy_type == "gaussian_actor":\n            architecture = "gaussian_actor"',
    '        elif policy_type == "gaussian_actor":\n            architecture = "gaussian_actor"\n        elif policy_type == "smolvla":\n            architecture = "smolvla"',
)
patch(
    modelcfg,
    '''            if architecture == "openvla_oft":
                from verl_vla.models.openvla_oft.processing_prismatic import PrismaticProcessor

                self.processor = PrismaticProcessor.from_pretrained(self.local_tokenizer_path)
                self.tokenizer = self.processor.tokenizer
            else:''',
    '''            if architecture == "openvla_oft":
                from verl_vla.models.openvla_oft.processing_prismatic import PrismaticProcessor

                self.processor = PrismaticProcessor.from_pretrained(self.local_tokenizer_path)
                self.tokenizer = self.processor.tokenizer
            elif architecture == "smolvla":
                self.tokenizer = None
                self.processor = None
            else:''',
)
print("registration patch done")
