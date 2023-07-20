import gradio as gr
import librosa
from asr import transcribe, ASR_EXAMPLES, ASR_LANGUAGES, ASR_NOTE
from tts import synthesize, TTS_EXAMPLES, TTS_LANGUAGES
from lid import identify, LID_EXAMPLES


demo = gr.Blocks()

mms_select_source_trans = gr.Radio(
    ["Record from Mic", "Upload audio"],
    label="Audio input",
    value="Record from Mic",
)
mms_mic_source_trans = gr.Audio(source="microphone", type="filepath", label="Use mic")
mms_upload_source_trans = gr.Audio(
    source="upload", type="filepath", label="Upload file", visible=False
)
mms_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        mms_select_source_trans,
        mms_mic_source_trans,
        mms_upload_source_trans,
        gr.Dropdown(
            [f"{k} ({v})" for k, v in ASR_LANGUAGES.items()],
            label="Language",
            value="eng English",
        ),
        # gr.Checkbox(label="Use Language Model (if available)", default=True),
    ],
    outputs="text",
    examples=ASR_EXAMPLES,
    title="Speech-to-text",
    description=(
        "Transcribe audio from a microphone or input file in your desired language."
    ),
    article=ASR_NOTE,
    allow_flagging="never",
)

mms_synthesize = gr.Interface(
    fn=synthesize,
    inputs=[
        gr.Text(label="Input text"),
        gr.Dropdown(
            [f"{k} ({v})" for k, v in TTS_LANGUAGES.items()],
            label="Language",
            value="eng English",
        ),
        gr.Slider(minimum=0.1, maximum=4.0, value=1.0, step=0.1, label="Speed"),
    ],
    outputs=[
        gr.Audio(label="Generated Audio", type="numpy"),
        gr.Text(label="Filtered text after removing OOVs"),
    ],
    examples=TTS_EXAMPLES,
    title="Text-to-speech",
    description=("Generate audio in your desired language from input text."),
    allow_flagging="never",
)

mms_select_source_iden = gr.Radio(
    ["Record from Mic", "Upload audio"],
    label="Audio input",
    value="Record from Mic",
)
mms_mic_source_iden = gr.Audio(source="microphone", type="filepath", label="Use mic")
mms_upload_source_iden = gr.Audio(
    source="upload", type="filepath", label="Upload file", visible=False
)
mms_identify = gr.Interface(
    fn=identify,
    inputs=[
        mms_select_source_iden,
        mms_mic_source_iden,
        mms_upload_source_iden,
    ],
    outputs=gr.Label(num_top_classes=10),
    examples=LID_EXAMPLES,
    title="Language Identification",
    description=("Identity the language of input audio."),
    allow_flagging="never",
)

tabbed_interface = gr.TabbedInterface(
    [mms_transcribe, mms_synthesize, mms_identify],
    ["Speech-to-text", "Text-to-speech", "Language Identification"],
)

with gr.Blocks() as demo:
    gr.Markdown(
        "<p align='center' style='font-size: 20px;'>MMS: Scaling Speech Technology to 1000+ languages demo. See our <a href='https://ai.facebook.com/blog/multilingual-model-speech-recognition/'>blog post</a> and <a href='https://arxiv.org/abs/2305.13516'>paper</a>.</p>"
    )
    gr.HTML(
        """<center>Click on the appropriate tab to explore Speech-to-text (ASR), Text-to-speech (TTS) and Language identification (LID) demos.   </center>"""
    )
    gr.HTML(
        """<center><a href="https://huggingface.co/spaces/facebook/MMS?duplicate=true"  style="display: inline-block;margin-top: .5em;margin-right: .25em;" target="_blank"><img style="margin-bottom: 0em;display: inline;margin-top: -.25em;" src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a> for more control and no queue.</center>"""
    )

    tabbed_interface.render()
    mms_select_source_trans.change(
        lambda x: [
            gr.update(visible=True if x == "Record from Mic" else False),
            gr.update(visible=True if x == "Upload audio" else False),
        ],
        inputs=[mms_select_source_trans],
        outputs=[mms_mic_source_trans, mms_upload_source_trans],
        queue=False,
    )
    mms_select_source_iden.change(
        lambda x: [
            gr.update(visible=True if x == "Record from Mic" else False),
            gr.update(visible=True if x == "Upload audio" else False),
        ],
        inputs=[mms_select_source_iden],
        outputs=[mms_mic_source_iden, mms_upload_source_iden],
        queue=False,
    )
    gr.HTML(
        """
            <div class="footer" style="text-align:center">
                <p>
                    Model by <a href="https://ai.facebook.com" style="text-decoration: underline;" target="_blank">Meta AI</a> - Gradio Demo by 🤗 Hugging Face
                </p>
            </div>
           """
        )

demo.queue(concurrency_count=3)
demo.launch()
