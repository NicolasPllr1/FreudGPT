import gradio as gr

from utils import (
    load_model,
    generate_answer,
    transcribe,
)

import constants as cst


def predict(
    message: str,
    history: list[list[str, str]],
    psy_name_chosen: str,
    model_name: str,
    # audio_recording,
):
    """
    Takes in a dialogue history and a message, and returns a response.

    Parameters:
        message: new message from the user.
        history: list of two-element lists containing the message-response history.
        psy_name_chosen: name of the psychoanalyst chosen. The response will be generated based on this persona.
        model_name: name of the model to use to generate the response. Default value or chosen by the user.
    """

    # If the user has recorded a question, we use it instead of the text input
    if cst.AUDIO_QUESTION is not None and len(cst.AUDIO_QUESTION) > 0:
        print(f"Audio question : {cst.AUDIO_QUESTION}")
        message = cst.AUDIO_QUESTION
        # Reset the audio question
        cst.AUDIO_QUESTION = None

    yield from generate_answer(
        psy_name_chosen,
        model_name,
        message,
        history,
        # model,
        # tokenizer,
    )


# Quel est le lien entre la pulsion auto-érotique et le narcissisme ? En prenant en compte l'évolution de ta pensée de 1905 à la fin de ta carrière ?
def gradio_app():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # ------------------
        # Psychologist selection
        psy_name_chosen = gr.Dropdown(
            choices=cst.PSY_NAMES,
            value=cst.DEFAULT_PSY,
            label="Choose Your Assistant",
        )

        # ------------------
        # Model selection
        model_name_chosen = gr.Dropdown(
            choices=cst.MODEL_NAMES,
            value=cst.DEFAULT_MODEL,
            label="Choose Your Model",
        )
        b_load_model = gr.Button("Load model")
        b_load_model.click(
            load_model,
            inputs=[model_name_chosen],
        )

        # ------------------
        # audio input
        audio_recording = gr.Audio(
            sources=["microphone", "upload"],
            label="Record your question",
        )
        b_transcribe_audio = gr.Button("Transcribe your vocal")
        b_transcribe_audio.click(
            transcribe,
            inputs=[audio_recording],
            show_progress=True,
        )

        # ------------------
        # chatbot interface
        chat = gr.ChatInterface(
            predict,
            additional_inputs=[psy_name_chosen, model_name_chosen],
            title="FreudGPT",
        )

    demo.queue().launch()


if __name__ == "__main__":
    # Launch the Gradio interface
    gradio_app()
