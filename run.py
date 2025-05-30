import shutil
import time
from datetime import datetime
from pathlib import Path
import ray

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from utils.util_fns import set_random_seeds, setup_logging
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="config",
            config_name='toydeceptive_pbt_00'
            )
def main(cfg: DictConfig) -> None:
    ray.init(address=cfg.server.ray_address)
    cfg.general.seed_base += cfg.general.seed_offset
    set_random_seeds(cfg.general.seed_base)

    exp_dir = Path(cfg.path.dir_exp)

    if cfg.general.continue_auto:
        exp_dir.mkdir(exist_ok=True, parents=True)
    else:
        shutil.rmtree(exp_dir, ignore_errors=True)
        exp_dir.mkdir(parents=True)

    setup_logging(exp_dir / '_log.txt')
    OmegaConf.save(cfg, exp_dir / 'config.yaml')

    # create file the name of which is cfg.general.exp_desc
    with open(exp_dir / cfg.general.exp_desc, 'w') as f:
        f.write('')

    ss = instantiate(cfg.search_space, seed=cfg.general.seed_base)
    task = instantiate(cfg.task, search_space=ss, cfg=cfg, _recursive_=False)

    algo = instantiate(cfg.algo, search_space=ss, cfg=cfg, task=task, _recursive_=False)

    st = time.time()
    algo.run()
    runtime = time.time() - st
    yaml.safe_dump(runtime, open(exp_dir / 'runtime.yaml', 'w'))

    print('Success', datetime.now().strftime('%H:%M:%S'))


if __name__ == '__main__':
    main()
