from openai import OpenAI
import gradio as gr

import logging

logger = logging.getLogger()

# /!\ You need your API secret key as an environment variable /!\
client = OpenAI()


def predict(message, history):
    psychologist_context = "You are Sigmund Freud, the father of modern psychanalysis. You will be helping a fellow psychologist. Please explain your ideas, concepts, and resonnings step by step when answering their questions. You can cite your work and books as references. Please answer in French."
    history_openai_format = [{"role": "system", "content": psychologist_context}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response_stream = client.chat.completions.create(
        model="gpt-3.5-turbo", messages=history_openai_format, stream=True
    )

    partial_message = ""
    for chunk in response_stream:
        if chunk.choices[0].delta.content is not None:
            partial_message = partial_message + chunk.choices[0].delta.content
            yield partial_message


# My patient always tell me how despicable her father is. How can I help her?
def gradio_app():
    gr.ChatInterface(
        predict,
        title="FreudGPT",
        theme=gr.themes.Soft(),
    ).queue().launch()


if __name__ == "__main__":
    # Launch the Gradio interface
    logger.info("Launching Gradio Interface...")
    gradio_app()
