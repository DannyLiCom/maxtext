import maxtext_trillium_model_configs as v6e_model_configs
import maxtext_v5e_model_configs as v5e_model_configs
import maxtext_v5p_model_configs as v5p_model_configs


AVAILABLE_MODELS_FRAMEWORKS = ["mcjax", "pathways"]

AVAILABLE_MODELS_NAMES = {
    'v6e': {
        'llama3_1_8b_8192': v6e_model_configs.llama3_1_8b_8192,
        'llama3_1_70b_8192': v6e_model_configs.llama3_1_70b_8192,
        # ... Other v6e models.
    },
    'v5litepod': {
        'llama2_13b_v5e_256': v5e_model_configs.llama2_13b_v5e_256,
        'gpt_3_175b_v5e_256': v5e_model_configs.gpt_3_175b_v5e_256,
        'llama2_70b_v5e_256': v5e_model_configs.llama2_70b_v5e_256,
        # ... Other v5e models.
    },
    'v5p': {
        'deepseek_v3_ep_256_v5p_512': v5p_model_configs.deepseek_v3_ep_256_v5p_512,
        # ... Other v5p models.
    },
}