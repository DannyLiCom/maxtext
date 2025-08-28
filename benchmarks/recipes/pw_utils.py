import typing

def build_user_models(
  selected_model_framework: typing.List[str],
  selected_model_names: typing.List[str],
  device_base_type: str,
  available_model_frameworks: typing.List[str],
  available_models: typing.Dict
) -> typing.Dict:
  """
  Validates user-selected model frameworks and names, then builds the final models dictionary.

  Args:
    selected_model_framework: A list of user-selected frameworks (e.g., 'pathways').
    selected_model_names: A list of user-selected model names.
    device_base_type: The base device type (e'g', 'v5e').
    available_model_frameworks: A list of all available frameworks.
    available_models: A dictionary mapping device types to available models.

  Returns:
    A dictionary containing the final model configurations.

  Raises:
    ValueError: If a selected framework or model name is not available.
  """
  # Iterate through the list of user-selected model frameworks, validating each one
  for model_framework in selected_model_framework:
    if model_framework not in available_model_frameworks:
      raise ValueError(
        f"Model framework '{model_framework}' not available. "
        f"Available model frameworks are: {list(available_model_frameworks)}"
      )
  
  # Initialize the model_set list to store the user's selected model configurations
  if device_base_type not in available_models:
    raise ValueError(
      f"Unknown device base type: {device_base_type}. "
      f"Original device type was: {device_base_type}"
    )

  # Iterate through the list of user-selected model names, validating each one
  for model_name in selected_model_names:
    if model_name not in available_models[device_base_type]:
      raise ValueError(
        f"Model name '{model_name}' not available for device type '{device_base_type}'. "
        f"Available model names are: {list(available_models[device_base_type].keys())}"
      )
  
  # Build the model configuration
  models = {}
  for model_framework in selected_model_framework:
    models[model_framework] = []
    for model_name in selected_model_names:
      models[model_framework].append(available_models[device_base_type][model_name])
      
  return models