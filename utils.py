import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from transformers import pipeline
import numpy as np

import time
from threading import Thread

from openai import OpenAI

import constants as cst


# --------------------------------------------
# functions to generate the response


def generate_answer(
    psy_name_chosen: str,
    model_name_chosen: str,
    message: str,
    history: list[list[str, str]],
    # model: AutoModelForCausalLM | None = None,
    # tokenizer: AutoTokenizer | None = None,
):
    """
    Root function for generating answers.
    Dispatches the generation to the right function based on the chosen model.

    Parameters:
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
        model_name_chosen: name of the model to use to generate the response. Default value or chosen by the user.
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        model: model to use to generate the response.
        tokenizer: tokenizer to use to generate the response.
    """

    if "parrot" in model_name_chosen:
        return generate_parrot(message, history, psy_name_chosen)
    elif "gpt" not in model_name_chosen:
        return generate_hf(message, history, psy_name_chosen, cst.MODEL, cst.TOKENIZER)
    else:
        return generate_openai(message, history, psy_name_chosen)


def generate_parrot(
    message: str,
    history: list[list[str, str]],
    psy_name_chosen: str,
):
    """
    For testing purposes. Will be removed in the future.
    Repeat the message like a parrot with an intro message based on the chosen psychoanalyst.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
    """
    # Repeats the message like a parrot
    to_say = (
        f"Hey! {psy_name_chosen} here. Sorry I didn't hear well, did you say: {message}"
    )

    for len_current_msg in range(len(to_say)):
        time.sleep(0.02)
        yield to_say[:len_current_msg]


def generate_hf(
    message,
    history,
    psy_name_chosen,
    model,
    tokenizer,
):
    """
    Generates an answer using Hugging-Face's transformers.
    Responds to the user's message based on the dialogue history and the chosen psychologist persona.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
        model: model to use to generate the response.
        tokenizer: tokenizer to use to generate the response.
    """

    psy_context = cst.CONTEXT[psy_name_chosen]
    system_context = ["<system>:" + psy_context, ""]
    dialogue_history_to_format = [system_context] + history + [[message, ""]]
    messages = "".join(
        [
            "".join(["\n<human>:" + item[0], "\n<bot>:" + item[1]])
            for item in dialogue_history_to_format
        ]
    )
    input_tokens = tokenizer(messages, return_tensors="pt").input_ids.cuda()
    streamer = TextIteratorStreamer(
        tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_tokens,
        streamer=streamer,
        **cst.GENERATION_CONFIG,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)

    print("Started generating text ...")
    t.start()
    partial_message = ""
    for new_token in streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message
    print("Answer generated :)")


def generate_openai(
    message: str,
    history: list[list[str, str]],
    psy_name_chosen: str,
):
    """
    Generates an answer using OpenAI's API.
    Responds to the user's message based on the dialogue history and the chosen psychologist persona.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
    """

    # /!\ You need your API secret key as an environment variable /!\ to use OpenAI's API
    client = OpenAI()

    ## test if an OpenAI API key is available as en environment variable
    # if client.api_key is None:
    #    print("No OpenAI API key found. OpenAI's API will not be used.")

    # add the context of the chosen psychoanalyst
    psy_context = cst.CONTEXT[psy_name_chosen]
    history_openai_format = [{"role": "system", "content": psy_context}]

    # add the conversation history
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})

    # add the user's last message
    history_openai_format.append({"role": "user", "content": message})

    # generate the response through openai's API
    response_stream = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=history_openai_format, stream=True
    )

    print("Started generating text ...")
    partial_message = ""
    for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message
    print("Answer generated :)")


# --------------------------------------------
# functions to load the model


def load_model(
    model_name: str,
    precision: str = cst.PRECISION,
    cache_dir: str = cst.DEFAULT_CACHE,
):
    """
    Loads the model and its tokenizer.

    Parameters:
        model_name: name of the model to load.
        precision: precision to load the model in (quantization). Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    if "gpt" or "parrot" in model_name:
        print(f"Loading {model_name}")
        cst.TOKENIZER = model_name
        cst.MODEL = model_name
        print(f"{model_name} loaded :)")

    else:
        print(f"Loading {model_name}")
        model_id = cst.MODEL_IDS[model_name]
        if precision == "4":
            print("Loading model in 4 bits")
            cst.TOKENIZER = AutoTokenizer.from_pretrained(model_id)
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            cst.MODEL = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map="auto",  # accelerate dispatches layers to ram, vram or disk
                cache_dir=cache_dir,
            )
            print(f"{model_name} loaded :)")


# --------------------------------------------
# functions to  transcribe the recorded audio

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")


def transcribe(audio: (int, np.ndarray)):
    """
    Transcribes the audio recording (speech2text)

    Parameters:
        audio: tuple containing the sampling rate and the audio recording.
    """
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    print("Transcribing the audio ...")
    cst.AUDIO_QUESTION = transcriber({"sampling_rate": sr, "raw": y})["text"]
    print("Transcription is done :) Got the following text:\n", cst.AUDIO_QUESTION)
