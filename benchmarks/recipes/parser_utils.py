import argparse

def add_arguments(parser: argparse.ArgumentParser):
  """Add arguments to arg parsers that need it.

  Args:
    parser:  parser to add shared arguments to.
  """
  # Add the arguments for each parser.
  # GCP Configuration
  parser.add_argument(
      '--user',
      type=str,
      default='user_name',
      help='GCP user name.')
  parser.add_argument(
      '--cluster_name',
      type=str,
      default='test-v5e-32-cluster',
      help='Name of the TPU cluster.')
  parser.add_argument(
      '--project',
      type=str,
      default='cloud-tpu-cluster',
      help='GCP project ID.')
  parser.add_argument(
      '--zone',
      type=str,
      default='us-south1-a',
      help='GCP zone for the cluster.')
  parser.add_argument(
      '--device_type',
      type=str,
      default='v5litepod-32',
      help='Type of TPU device (e.g., v5litepod-32).')
  parser.add_argument(
      '--priority',
      type=str,
      choices=['low', 'medium', 'high', 'very high'],
      default='medium',
      help='Priority of the job.')

  # Image Configuration
  parser.add_argument(
      '--server_image',
      type=str,
      default='us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server',
      help='Docker image for the proxy server.')
  parser.add_argument(
      '--proxy_image',
      type=str,
      default='us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server',
      help='Docker image for the server.')
  parser.add_argument(
      '--runner',
      type=str,
      default='us-docker.pkg.dev/path/to/maxtext_runner',
      help='Docker image for the runner.')
  parser.add_argument(
      '--colocated_python_image',
      type=str,
      default=None,
      help='Colocated Python image.')
  parser.add_argument(
      '--worker_flags',
      type=str,
      default='',
      help='Worker flags.')
  parser.add_argument(
      '--proxy_flags',
      type=str,
      default='',
      help='Proxy flags.')
  parser.add_argument(
      '--server_flags',
      type=str,
      default='',
      help='Server flags.')

  # Model Configuration
  parser.add_argument(
      '--benchmark_steps',
      type=int,
      default=20,
      help='Number of benchmark steps.')
  parser.add_argument(
      '--headless',
      action=argparse.BooleanOptionalAction,
      default=False,
      help='Run in headless mode.')
  parser.add_argument(
      '--selected_model_framework',
      nargs='+',
      default=['pathways'],
      help='List of model frameworks (e.g., pathways mcjax).')
  parser.add_argument(
      '--selected_model_names',
      nargs='+',
      default=['llama3_1_8b_8192_v5e_256'],
      help='List of model names (e.g., llama3_1_8b_8192_v5e_256).')
  parser.add_argument(
      '--num_slices_list',
      nargs='+',
      type=int,
      default=[2],
      help='List of number of slices.')

  # Other configurations
  parser.add_argument(
      '--xpk_path',
      type=str,
      default='~/xpk',
      help='Path to xpk.')
  parser.add_argument(
      '--delete',
      action='store_true',
      help='Delete the cluster workload')
  parser.add_argument(
      '--max_restarts',
      type=int,
      default=0,
      help='Maximum number of restarts')

# def main():
#     parser = argparse.ArgumentParser(description="Main script description.")
#     add_arguments(parser)
#     args = parser.parse_args()
#     print(f"User: {args.user}")
#     print(f"Cluster Name: {args.cluster_name}")
#     print(f"Selected Model Names: {args.selected_model_names}")

# if __name__ == '__main__':
#     main()