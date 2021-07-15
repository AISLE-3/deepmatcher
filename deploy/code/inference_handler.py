import os
import sys
import json
import torch
import torch.nn.functional as F
import logging
import pandas as pd

from deepmatcher.models.core import MatchingModel
from deepmatcher.data.custom_dataset import DeepMatcherDataset
from deepmatcher.data.encoder.text_encoders import HFTextEncoder
from deepmatcher.data.encoder.image_encoders import DINOImageEncoder
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors


CONFIG = {
    "text_encoder_identifier" : "albert-base-v2",
    "text_encoder_trainable" : False,
    "text_encoder_seq_to_vec" : "cls",
    "text_attrs" : ['title'],
    "image_attr" : '',
    "prefixes" : ["left_", "right_"]
}

print("defined config:")

class DMInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_dir):
        print('executing model_fn')
        print('model_dir', model_dir)
        print('model_dir_contents:', os.listdir(model_dir))
        try:
            self.text_encoder_identifier = CONFIG["text_encoder_identifier"]
            self.text_encoder_trainable = CONFIG['text_encoder_trainable']
            self.text_encoder_seq_to_vec = CONFIG['text_encoder_seq_to_vec']
            self.text_encoder = HFTextEncoder(self.text_encoder_identifier, trainable=self.text_encoder_trainable, seqtovec=self.text_encoder_seq_to_vec)
            self.tokenizer = self.text_encoder.tokenizer
            # self.image_encoder = DINOImageEncoder()
            self.image_encoder = None

            model = MatchingModel.load_from_state(
                os.path.join(model_dir, model_path),
                text_encoder=self.text_encoder,
                image_encoder=self.image_encoder,
            )
            print(model)
            model = model.eval()
            return model
        except Exception as e:
            logging.exception(e)

    def default_input_fn(self, input_data, content_type):
        """
        Args:
            request_body (str): Expecting this to be a json document that contains elements of pairs that are to be matched together, eg: [{left_id: "offer_id_666", left_title: "some title text", right_id: "offer_id_555", right_title: "more title text"},{..more..} ..]
            request_content_type (str): Expecting this to be application/json
        """
        try:
            print('input_fn:', 'input_data', input_data, 'type', type(input_data), 'content_type', content_type)
            input_data = decoder.decode(input_data, content_type)
            print('decoded input data:', input_data, 'type', type(input_data))
            # input_data = input_data.item()
            input_data = input_data.tolist()
            print('unpickled dtype:', type(input_data))
            if isinstance(input_data, str):
                input_data = json.loads(input_data)
            if isinstance(input_data, dict):
                input_data = [input_data]
            assert isinstance(input_data, list), f"expected dtype for input data : list but got : {type(input_data)}"
            assert isinstance(input_data[0], dict), f"expected dtype for iterable elements : dict but got : {type(input_data)}"

            data_df = pd.DataFrame(input_data)
            dataset =  DeepMatcherDataset(
                data_df=data_df,
                label_col=None,
                text_cols=CONFIG["text_attrs"],
                image_col=CONFIG["image_attr"],
                tokenizer=self.tokenizer,
                prefixes=CONFIG["prefixes"],
            )
            batch = [dataset[i] for i in range(len(dataset))]
            batch = dataset.collate_fn(batch)
            return batch['attrs']
        except Exception as e:
            logging.exception(e)

    def default_predict_fn(self, input_object, model):
        try:
            print('predict_fn', input_data)
            logits = model(input_data)
            print('logits', logits)
            preds = F.softmax(logits, dim=1)
            matches = torch.argmax(preds, dim=1)
            matches = matches.flatten().cpu().numpy()
            print('matches', matches)
            return matches
        except Exception as e:
            logging.exception(e)

    def default_output_fn(self, prediction, content_type):
        try:
            return encoder.encode({"results" : list(prediction)}, content_type)
        except Exception as e:
            logging.exception(e)
