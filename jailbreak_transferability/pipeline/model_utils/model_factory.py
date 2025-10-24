from pipeline.model_utils.model_base import ModelBase

def construct_model_base(model_path: str) -> ModelBase:
    if 'llama-2' in model_path.lower():
        from pipeline.model_utils.models.llama2_model import Llama2Model
        return Llama2Model(model_path)
    if 'llama-3' in model_path.lower():
        from pipeline.model_utils.models.llama3_2_model import Llama3_2Model
        return Llama3_2Model(model_path)
    if 'qwen2.5' in model_path.lower():
        from pipeline.model_utils.models.qwen2_5_model import Qwen2_5Model
        return Qwen2_5Model(model_path)
    if 'vicuna' in model_path.lower():
        from pipeline.model_utils.models.vicuna_model import VicunaModel
        return VicunaModel(model_path)