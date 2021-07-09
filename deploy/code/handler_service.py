from inference_handler import DMInferenceHandler
from sagemaker_inference.transformer import Transformer
from sagemaker_inference.default_handler_service import DefaultHandlerService


class HandlerService(DefaultHandlerService):
    def __init__(self):
        transformer = Transformer(default_inference_handler=DMInferenceHandler)
        super(HandlerService, self).__init__(transformer=transformer)
