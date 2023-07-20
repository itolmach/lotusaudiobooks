from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
import torch
import librosa

model_id = "facebook/mms-lid-1024"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)


LID_SAMPLING_RATE = 16_000
LID_TOPK = 10
LID_THRESHOLD = 0.33

LID_LANGUAGES = {}
with open(f"data/lid/all_langs.tsv") as f:
    for line in f:
        iso, name = line.split(" ", 1)
        LID_LANGUAGES[iso] = name


def identify(audio_source=None, microphone=None, file_upload=None):
    if audio_source is None and microphone is None and file_upload is None:
        # HACK: need to handle this case for some reason
        return {}

    if type(microphone) is dict:
        # HACK: microphone variable is a dict when running on examples
        microphone = microphone["name"]
    audio_fp = (
        file_upload if "upload" in str(audio_source or "").lower() else microphone
    )
    if audio_fp is None:
        return "ERROR: You have to either use the microphone or upload an audio file"
    
    audio_samples = librosa.load(audio_fp, sr=LID_SAMPLING_RATE, mono=True)[0]

    inputs = processor(
        audio_samples, sampling_rate=LID_SAMPLING_RATE, return_tensors="pt"
    )

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    ):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model.to(device)
    inputs = inputs.to(device)

    with torch.no_grad():
        logit = model(**inputs).logits

    logit_lsm = torch.log_softmax(logit.squeeze(), dim=-1)
    scores, indices = torch.topk(logit_lsm, 5, dim=-1)
    scores, indices = torch.exp(scores).to("cpu").tolist(), indices.to("cpu").tolist()
    iso2score = {model.config.id2label[int(i)]: s for s, i in zip(scores, indices)}
    if max(iso2score.values()) < LID_THRESHOLD:
        return "Low confidence in the language identification predictions. Output is not shown!"
    return {LID_LANGUAGES[iso]: score for iso, score in iso2score.items()}


LID_EXAMPLES = [
    [None, "./assets/english.mp3", None],
    [None, "./assets/tamil.mp3", None],
    [None, "./assets/burmese.mp3", None],
]
