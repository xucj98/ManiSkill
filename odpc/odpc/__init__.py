from omegaconf import OmegaConf

if not OmegaConf.has_resolver("fmt"):
    OmegaConf.register_new_resolver("fmt", lambda v, f: f"{v:{f}}")
