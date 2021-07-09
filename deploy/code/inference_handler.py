import os
import sys
import json
import torch
import logging
import pandas as pd
from torch.utils.data import DataLoader
from deepmatcher.models.core import MatchingModel
from deepmatcher.data.custom_dataset import DeepMatcherDataset
from deepmatcher.data.encoder.text_encoders import HFTextEncoder
from sagemaker_inference import content_types, decoder, default_inference_handler, encoder, errors

class DMInferenceHandler(default_inference_handler.DefaultInferenceHandler):

    def default_model_fn(self, model_path):

        try:
            model = MatchingModel(
                # text_encoder=text_encoder,
                # image_encoder=image_encoder,
                attr_summarizer=None,
            )
            model.load_state(model_path)
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

            text_encoder = HFTextEncoder('sentence-transformers/stsb-mpnet-base-v2', trainable=False, seqtovec="cls")

            # I imagine request_body is going to be a string, parse it
            data = json.loads(request_body)
            data_df = pd.read_json(data)

            return DeepMatcherDataset(
                data=data_df,
                label_col='label',
                text_columns=['title'],
                image_col=''
                _tokenizer=text_encoder.tokenizer
            )

        except Exception as e:
            logging.exception(e)


    def default_predict_fn(self, input_object, model):
        try:
            predictions = model.run_prediction(
                            input_object,
                            output_attributes=[
                                "left_id",
                                "right_id",
                            ])
        except Exception as e:
            logging.exception(e)


    def default_output_fn(self, prediction, response_content_type):
        try:
            return prediction.to_json()
        except Exception as e:
            logging.exception(e)
