# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""chatlearn launcher"""

import argparse
from importlib import import_module
import sys
import traceback
from typing import Dict, Tuple, Type, Any
from omegaconf import OmegaConf
import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.config_store import ConfigStore

from chatlearn.algorithm.base_algo import BaseAlgorithm
from chatlearn.utils.parse_utils import find_parser_from_keyname

# e.g. python chatlearn/chatlearn.py grpo --config-file grpo.yaml runtime.data_path=/tmp/data runtime.eval_data_path=/tmp/eval_data

# Registry format:
#  "engine_name": ("module_path", "algo_class_name", "config_class")
ALGO_REGISTRY: Dict[str, Tuple[str, str, str]] = {
    "grpo": ("algorithm.grpo", "GrpoAlgorithm", "GrpoConfig"),
}


class ChatlearnLauncher:
    """ChatlearnLauncher"""

    def __init__(self) -> None:
        self.parser = self._create_parser()


    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="ChatLearn: An RLHF Training System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            # add_help=False,
        )

        subparsers = parser.add_subparsers(
            title="Available algorithms",
            dest="algorithm",
            metavar="ALGORITHM"
        )

        for algo_name in ALGO_REGISTRY:
            algo_parser = subparsers.add_parser(
                algo_name,
                description=f"Run {algo_name.upper()} algorithm",
                help=f"{algo_name.upper()} algorithm",
                formatter_class=argparse.RawDescriptionHelpFormatter,
            )
            algo_parser.add_argument(
                "--config-file",
                type=str,
                help="Path to the config file",
            )
            algo_parser.add_argument(
                "hydra_args",
                nargs=argparse.REMAINDER,
                help="Hydra configs (e.g. ++key=value)"
            )

        return parser


    def _load_algorithm(self, algo_name: str) -> Tuple[Type[BaseAlgorithm], Type[Any]]:
        module_path, algo_cls_name, config_cls_name = ALGO_REGISTRY[algo_name]
        try:
            module = import_module(module_path)
            algo_cls = getattr(module, algo_cls_name)
            config_cls = getattr(module, config_cls_name)
            return algo_cls, config_cls
        except Exception as e:
            raise RuntimeError(f"Failed to load algorithm: {algo_name} ({str(e)})") from e


    def _run_algorithm(self, algo_args) -> None:
        algo_cls, config_cls = self._load_algorithm(algo_args.algorithm)
        cs = ConfigStore.instance()
        cs.store(name=algo_args.algorithm, node=config_cls)
        GlobalHydra.instance().clear()
        with hydra.initialize(config_path=None, version_base=None):
            cfg = hydra.compose(config_name=algo_args.algorithm)
            if algo_args.config_file is not None:
                external_cfg = OmegaConf.load(algo_args.config_file)
                keynames, values = list(zip(*[arg.split('=', 1) for arg in algo_args.hydra_args if '=' in arg]))
                default_merged_config = OmegaConf.to_object(OmegaConf.merge(cfg, external_cfg))
                parsers = find_parser_from_keyname(default_merged_config, keynames)
                for keyname, value in zip(keynames, values):
                    parser = parsers[keyname]
                    if parser is not None:
                        value = parser(value)
                    OmegaConf.update(external_cfg, keyname, value)
                cfg = OmegaConf.merge(cfg, external_cfg) # include $
            cfg = OmegaConf.to_object(cfg) # real cfg from template and user input
            instance = algo_cls(cfg) # algo may update cfg
            instance.validate()
            instance.run()


    def run(self) -> None:
        args, _ = self.parser.parse_known_args()

        if not args.algorithm:
            self.parser.print_help()
            return

        if args.algorithm not in ALGO_REGISTRY:
            print(f"ERROR: Unknown algorithm {args.algorithm}")
            self.parser.print_help()
            sys.exit(1)

        algo_args = self.parser.parse_args()

        try:
            self._run_algorithm(algo_args)
        except Exception:
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    launcher = ChatlearnLauncher()
    launcher.run()
    