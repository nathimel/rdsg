from omegaconf import DictConfig, OmegaConf
import hydra
import os

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def myfunc(cfg):
    print(OmegaConf.to_yaml(cfg))

    filepaths = cfg.filepaths
    fn = filepaths.tradeoff_plot_fn
    cwd = os.getcwd()

    fn = os.path.join(cwd, fn)

    with open(fn, 'w') as f:
        f.write("hi")

if __name__ == "__main__":
    myfunc()
