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

class DMInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    def default_model_fn(self, model_path):
        try:
            self.text_encoder_identifier = os.environ["text_encoder_identifier"]
            self.text_encoder_trainable = os.environ['text_encoder_trainable']
            self.text_encoder_seq_to_vec = os.environ['text_encoder_seq_to_vec']
            self.text_encoder = HFTextEncoder(self.text_encoder_identifier, trainable=self.text_encoder_trainable, seqtovec=self.text_encoder_seq_to_vec)
            self.image_encoder = DINOImageEncoder()

            model = MatchingModel.load_from_state(
                model_path,
                text_encoder=self.text_encoder,
                image_encoder=self.image_encoder,
            )
            model = model.eval()
            return model
        except Exception as e:
            logging.exception(e)

    def default_input_fn(self, request_body, request_content_type):
        """
        Args:
            request_body (str): Expecting this to be a json document that contains elements of pairs that are to be matched together, eg: [{left_id: "offer_id_666", left_title: "some title text", right_id: "offer_id_555", right_title: "more title text"},{..more..} ..]
            request_content_type (str): Expecting this to be application/json
        """
        try:
            logging.debug("request_body_type: {}, request_body: {}".format(type(request_body), request_body))
            # I imagine request_body is going to be a string, parse it
            # data = json.loads(request_body)
            data_df = pd.read_json(request_body)
            dataset =  DeepMatcherDataset(
                data_df=data_df,
                label_col=None,
                text_cols=self.model.text_attrs,
                image_col=self.model.image_attr,
                tokenizer=self.text_encoder.tokenizer,
                prefixes=self.model.prefixes
            )
            batch = [dataset[i] for i in range(len(dataset))]
            batch = dataset.collate_fn(batch)
            return batch['attrs']
        except Exception as e:
            logging.exception(e)

    def default_predict_fn(self, input_object, model):
        try:
            logits = model(input_object)
            preds = F.softmax(logits, dim=1)
            matches = torch.argmax(preds, dim=1)
            matches = matches.flatten().cpu().numpy()
            return matches
        except Exception as e:
            logging.exception(e)

    def default_output_fn(self, prediction, response_content_type):
        try:
            return {"results" : list(prediction)}
        except Exception as e:
            logging.exception(e)
