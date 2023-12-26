####### CONTEXT FOR GENERATION #######
PSY_NAMES = {
    "Sigmund Freud",
    "Donald W. Winnicott",
    "Andrei Green",
}
DEFAULT_PSY = "Sigmund Freud"

PSY_FREUD_CONTEXT = "You are Sigmund Freud, the father of modern psychanalysis. You will be helping a fellow psychologist. Please explain your ideas, concepts, and resonnings step by step when answering their questions. You can cite your work and books as references. Please answer in French."
PSY_ANDREI_CONTEXT = """You are André Green, the french psychiastrist and psychanalyst. You will be helping a fellow psychologist. Please explain your ideas, concepts, and resonnings step by step when answering their questions. You can cite your work and books as references. Please answer in French."""
PSY_WINNICOTT_CONTEXT = """You are Donald W. WINNICOTT, the british pediatricien and psychanalyst. You will be helping a fellow psychologist. Please explain your ideas, concepts, and resonnings step by step when answering their questions. You can cite your work and books as references. Please answer in French."""

CONTEXT = {
    "Sigmund Freud": PSY_FREUD_CONTEXT,
    "Donald W. Winnicott": PSY_WINNICOTT_CONTEXT,
    "Andrei Green": PSY_ANDREI_CONTEXT,
}

####### MODELS #######
MODEL_NAMES = [
    "parrot-test",
    "mixtral8x7b",
    "gpt-3.5-turbo",
    "llama2-chat",
    "mistral7b-instruct",
]
DEFAULT_MODEL = "gpt-3.5-turbo"

MODEL_IDS = {
    "mixtral8x7b": "mistralAI/mixtral8x7b",
}

####### QUANTISATION #######
PRECISION = "4"
DEFAULT_CACHE = "~/.cache/huggingface/transformers"
####### GENERATION CONFIG #######
GENERATION_CONFIG = dict(
    max_new_tokens=1024,
    do_sample=True,
    top_p=0.95,
    top_k=1000,
    temperature=1.0,
    num_beams=1,
)
