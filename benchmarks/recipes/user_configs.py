# Copyright 2023–2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import maxtext_xpk_runner as mxr
from xpk_configs import XpkClusterConfig
import maxtext_trillium_model_configs as v6e_model_configs
import maxtext_v5e_model_configs as v5e_model_configs
import maxtext_v5p_model_configs as v5p_model_configs

@dataclasses.dataclass
class UserConfig:
  """The default configuration can be modified here."""
  # gcp configuration
  user: str = 'user_name'
  cluster_name: str = 'test-v5e-32-cluster'
  project: str = 'cloud-tpu-cluster'
  zone: str = 'us-south1-a'
  device_type: str = 'v5litepod-32'

  # Images for env
  server_image: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest'
  proxy_image: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest'
  runner: str = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/maxtext_jax_stable:latest'
  colocated_python_image: str = None

  # model configuration
  benchmark_steps: int = 20
  num_slices_list: list[int] = dataclasses.field(default_factory=lambda: [2])
  models: dict = dataclasses.field(default_factory=lambda: {
    'mcjax': [],
    'pathways': [
        v6e_model_configs.llama3_1_8b_8192,
    ],
  })
  
  xpk_path: str = '~/xpk'

  def __post_init__(self):
    """Automatically generate derived attributes after the object is created."""
    self.cluster_config = XpkClusterConfig(
        cluster_name=self.cluster_name,
        project=self.project,
        zone=self.zone,
        device_type=self.device_type,
    )
    
    self.region = '-'.join(self.zone.split('-')[:-1])
    self.headless = True

    self.pathways_config = mxr.PathwaysConfig(
        server_image=self.server_image,
        proxy_server_image=self.proxy_image,
        runner_image=self.runner,
        colocated_python_sidecar_image=self.colocated_python_image,
        headless=self.headless,
    )
    self.headless_workload_name = f'{self.user[:3]}-headless'
    self.base_output_directory = f'gs://{self.user}-{self.region}/{self.user}-'

if __name__ == '__main__':
  user_config = UserConfig()

  # Access the generated attributes
  for key, value in user_config.__dict__.items():
    print(f'{key}: {value}')
