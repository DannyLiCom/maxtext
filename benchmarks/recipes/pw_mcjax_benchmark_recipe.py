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
from . import new_args_helper as helper
# from . import user_configs
from .user_configs import UserConfig
from .user_configs import USER_CONFIG
from .runner_utils import generate_and_run_workloads
from . import parser_utils
import argparse

def main(user_config: UserConfig) -> int:
  """Main program entry point"""
  # user_configs.USER_CONFIG.headless = False


  # should_continue = helper.handle_cmd_args(
  #     user_config.cluster_config, helper.DELETE, xpk_path=user_config.xpk_path
  # )

  # if not should_continue:
  #   return 0

  # generate_and_run_workloads(user_configs.USER_CONFIG, user_configs.USER_CONFIG.num_slices_list, user_configs.USER_CONFIG.benchmark_steps, user_configs.USER_CONFIG.priority)
  generate_and_run_workloads(user_config, user_config.num_slices_list, user_config.benchmark_steps, user_config.priority)

  return 0


if __name__ == "__main__":
  # 第 1 步：解析所有命令列參數
  parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  parser_utils.add_arguments(parser)
  args = parser.parse_args()

  # 第 2 步：根據 sys.argv 的長度來決定執行路徑
  if len(sys.argv) > 2:
    # 模式 1: 傳入多個參數 (>2個)，屬於自動化測試情境
    print("Multiple command line arguments detected. Custom configuration will be used.")
    user_config = UserConfig(**vars(args))
    should_continue = helper.handle_cmd_args(
        user_config.cluster_config,
        is_delete=user_config.delete,
        user=user_config.user,
        xpk_path=user_config.xpk_path
    )
    if not should_continue:
      sys.exit(0)

    print(f"configuration used：{user_config}")
    main(user_config)
  
  # elif len(sys.argv) == 2 and "--delete" in sys.argv:
  #   # 模式 2: 僅傳入 --delete 參數 (2個參數)，屬於手動刪除情境
  #   print("The --delete flag is detected for manual deletion. Deletion will be performed.")
  #   user_config = USER_CONFIG
    
  #   should_continue = helper.handle_cmd_args(
  #       user_config.cluster_config,
  #       is_delete=user_config.delete,
  #       xpk_path=user_config.xpk_path
  #   )
  #   if not should_continue:
  #     sys.exit(0)
  
  else: # len(sys.argv) == 1
    # 模式 3: 無任何參數，屬於手動測試情境
    print("No command line arguments detected. The default configuration will be used.")
    user_config = USER_CONFIG
    if "--delete" in sys.argv:
      user_config.delete = True
      should_continue = helper.handle_cmd_args(
          user_config.cluster_config,
          is_delete=user_config.delete,
          user=user_config.user,
          xpk_path=user_config.xpk_path
      )
      if not should_continue:
        sys.exit(0)    
    print(f"configuration used：{user_config}")
    main(user_config)

  # # Step 1: Parse all command line parameters
  # parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  # parser_utils.add_arguments(parser)
  # args = parser.parse_args()

  # # Step 2: Create a UserConfig object using all parsed parameters
  # user_config = UserConfig(**vars(args))

  # # Step 3: Determine the execution path based on whether the --delete flag is present
  # if args.delete:
  #   print("Detects the --delete flag. Delete will be performed.")
    
  #   should_continue = helper.handle_cmd_args(
  #       user_config.cluster_config,  
  #       helper.DELETE, 
  #       xpk_path=user_config.xpk_path
  #   )
    
  #   if not should_continue:
  #     sys.exit(0)

  # # Phase 2: Handling two cases without the --delete flag
  # if len(sys.argv) > 1:
  #   # Case 1: There are other parameters (not --delete)
  #   parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  #   parser_utils.add_arguments(parser)
  #   args = parser.parse_args()
  #   user_config = UserConfig(**vars(args))
    
  #   print("Command line arguments detected. Custom configuration will be used")
  #   print(f"configuration used：{user_config}")
  #   main(user_config)
  # else:
  #   # Case 2: No parameters
  #   user_config = UserConfig()
    
  #   print("No command line arguments were detected. The default configuration will be used")
  #   print(f"configuration used：{user_config}")
  #   main(user_config)

  # parser = argparse.ArgumentParser(description="Used to perf benchmarks between Pathways and McJax.")
  # parser_utils.add_arguments(parser)
  # args = parser.parse_args()
  # if len(sys.argv) > 1:
  #   user_config = UserConfig(**vars(args))
  #   print(f"將使用的配置: {user_config}")
  #   main(user_config)
  #   # main()
  # else:
  #   user_config = UserConfig()
  #   main(user_config)