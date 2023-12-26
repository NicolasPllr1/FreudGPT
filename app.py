import gradio as gr

from utils import (
    load_model,
    generate_answer,
)

from constants import (
    PSY_NAMES,
    DEFAULT_PSY,
    MODEL_NAMES,
    DEFAULT_MODEL,
)


def predict(
    message: str,
    history: list[list[str, str]],
    psy_name_chosen: str,
    model_name: str,
):
    """
    Takes in a dialogue history and a message, and returns a response.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
        model_name: name of the model to use to generate the response. Default value or chosen by the user.
    """
    global model, tokenizer
    yield from generate_answer(
        psy_name_chosen,
        model_name,
        message,
        history,
        model,
        tokenizer,
    )


# Quel est le lien entre la pulsion auto-érotique et le narcissisme ? En prenant en compte l'évolution de ta pensée de 1905 à la fin de ta carrière ?
def gradio_app():
    global model, tokenizer
    tokenizer, model = None, None

    with gr.Blocks() as demo:
        psy_name_chosen = gr.Dropdown(
            choices=PSY_NAMES,
            value=DEFAULT_PSY,
            label="Choose Your Assistant",
        )

        model_name_chosen = gr.Dropdown(
            choices=MODEL_NAMES,
            value=DEFAULT_MODEL,
            label="Choose Your Model",
        )
        b = gr.Button("Load model")
        b.click(
            load_model,
            inputs=[model_name_chosen],
        )

        gr.ChatInterface(
            predict,
            additional_inputs=[psy_name_chosen, model_name_chosen],
            title="FreudGPT",
            theme="soft",
        )

    demo.queue().launch()


if __name__ == "__main__":
    # Launch the Gradio interface
    gradio_app()