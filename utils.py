import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
import time
from threading import Thread

from openai import OpenAI

from constants import CONTEXT, MODEL_IDS, PRECISION, GENERATION_CONFIG, DEFAULT_CACHE

# /!\ You need your API secret key as an environment variable /!\ to use OpenAI's API
client = OpenAI()

## test if an OpenAI API key is available as en environment variable
# if client.api_key is None:
#    print("No OpenAI API key found. OpenAI's API will not be used.")


def generate_answer(
    psy_name_chosen: str,
    model_name_chosen: str,
    message: str,
    history: list[list[str, str]],
    model: AutoModelForCausalLM | None = None,
    tokenizer: AutoTokenizer | None = None,
):
    """
    Root function for generating answers

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
        return generate_hf(message, history, psy_name_chosen, model, tokenizer)
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
    intro = f"Hey! {psy_name_chosen} there. Sorry I didn't hear well, did you say: "
    to_say = intro + message

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

    psy_context = CONTEXT[psy_name_chosen]
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
        **GENERATION_CONFIG,
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

    # add the context of the chosen psychoanalyst
    psy_context = CONTEXT[psy_name_chosen]
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


def load_model(
    model_name: str,
    precision: str = PRECISION,
    cache_dir: str = DEFAULT_CACHE,
):
    """
    Loads the model and its tokenizer.

    Parameters:
        model_name: name of the model to load.
        precision: precision to load the model in (quantization). Default value or chosen by the user.
        cache_dir: path to the cache directory. Default value.
    """

    global tokenizer, model

    if "gpt" or "parrot" in model_name:
        print(f"Loading {model_name}")
        tokenizer = model_name
        model = model_name

    else:
        print(f"Loading {model_name}")
        model_id = MODEL_IDS[model_name]
        if precision == "4":
            print("Loading model in 4 bits")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=nf4_config,
                device_map="auto",  # accelerate dispatches layers to ram, vram or disk
                cache_dir=cache_dir,
            )
