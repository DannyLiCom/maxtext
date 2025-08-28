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

"""Used to perf benchmarks between Pathways and McJax."""
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)
from . import args_helper as helper
from .user_configs import UserConfig
from .runner_utils import generate_and_run_workloads

def main() -> int:
  """Main program entry point"""
  # Define user specific configurations for recipes here
  user_config = UserConfig(
      user='lidanny',
      cluster_name='pw-scale-test-v5e-32',
      project='cloud-tpu-multipod-dev',
      zone='us-south1-a',
      device_type='v5litepod-32',
      benchmark_steps=20,
      num_slices_list=[2],
      server_image = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_server:latest',
      proxy_image = 'us-docker.pkg.dev/cloud-tpu-v2-images-dev/pathways/unsanitized_proxy_server:latest',
      runner='gcr.io/tpu-prod-env-one-vm/chzheng_latest:latest',
      selected_model_framework=['pathways'],
      selected_model_names=['llama3_1_8b_8192_v5e_256']
  )
  should_continue = helper.handle_cmd_args(
      user_config.cluster_config, helper.DELETE, xpk_path=user_config.xpk_path
  )

  if not should_continue:
    return 0

  generate_and_run_workloads(user_config, user_config.num_slices_list, user_config.benchmark_steps)

  return 0


if __name__ == "__main__":
  main()
