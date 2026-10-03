import base64
import io
import mimetypes
import random
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Waste Sorting Game",
    page_icon="♻️",
    layout="wide",
)

# =========================================================
# PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "efficientnet_b0_waste.pth"

GAME_IMAGES_DIR = BASE_DIR / "game_images"

ORGANIC_BIN_PATH = BASE_DIR / "assets" / "organic_bin.png"
INORGANIC_BIN_PATH = BASE_DIR / "assets" / "inorganic_bin.png"

# =========================================================
# CONFIG
# =========================================================

TOTAL_QUESTIONS = 10

CLASS_NAMES = [
    "Anorganik",
    "Organik",
]

# ImageNet normalization
game_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    model = efficientnet_b0(weights=None)

    # Same classifier architecture used during training
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        2
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model


model = load_model()


# =========================================================
# LOAD GAME IMAGES
# =========================================================

@st.cache_data
def get_game_images():

    images = []

    for class_name in ["Organik", "Anorganik"]:

        class_dir = GAME_IMAGES_DIR / class_name

        if not class_dir.exists():
            continue

        for image_path in class_dir.iterdir():

            if image_path.is_file() and image_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
                ".bmp"
            }:

                images.append({
                    "path": str(image_path),
                    "label": class_name,
                    "name": image_path.name
                })

    return images


game_images = get_game_images()

if len(game_images) == 0:
    st.error(
        "Tidak ada gambar game ditemukan. "
        "Pastikan folder game_images/Organik dan game_images/Anorganik sudah benar."
    )
    st.stop()


# =========================================================
# CONVERT IMAGE TO DATA URI
# =========================================================

def image_to_data_uri(
    image_path,
    max_size=700
):

    image = Image.open(image_path)

    # Pertahankan alpha/transparansi
    if image.mode != "RGBA":
        image = image.convert("RGBA")

    image.thumbnail((max_size, max_size))

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# CACHE BIN IMAGES
# =========================================================

@st.cache_data
def get_bin_images():

    organic = image_to_data_uri(
        ORGANIC_BIN_PATH,
        max_size=300
    )

    inorganic = image_to_data_uri(
        INORGANIC_BIN_PATH,
        max_size=300
    )

    return organic, inorganic


organic_bin_uri, inorganic_bin_uri = get_bin_images()


# =========================================================
# AI PREDICTION
# =========================================================

@st.cache_data
def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    tensor = game_transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_index = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_index
        ].item()

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# =========================================================
# RANDOM IMAGE
# =========================================================

def choose_random_image(exclude_path=None):

    candidates = [
        item
        for item in game_images
        if item["path"] != exclude_path
    ]

    if not candidates:
        candidates = game_images

    return random.choice(candidates)


# =========================================================
# SESSION STATE
# =========================================================

if "score" not in st.session_state:
    st.session_state.score = 0

if "question" not in st.session_state:
    st.session_state.question = 1

if "current_image" not in st.session_state:
    selected = choose_random_image()

    st.session_state.current_image = selected
    st.session_state.ai_prediction = predict_image(
        selected["path"]
    )

if "feedback" not in st.session_state:
    st.session_state.feedback = ""

if "feedback_type" not in st.session_state:
    st.session_state.feedback_type = ""

if "last_drop_id" not in st.session_state:
    st.session_state.last_drop_id = None

if "game_over" not in st.session_state:
    st.session_state.game_over = False

if "game_nonce" not in st.session_state:
    st.session_state.game_nonce = 0

# =========================================
# AI RESULT FOR LAST ANSWERED IMAGE
# =========================================

if "last_answered_image" not in st.session_state:
    st.session_state.last_answered_image = None

if "last_ai_prediction" not in st.session_state:
    st.session_state.last_ai_prediction = None

if "last_ai_confidence" not in st.session_state:
    st.session_state.last_ai_confidence = None


# =========================================================
# CURRENT IMAGE DATA
# =========================================================

current_image = st.session_state.current_image

current_image_uri = image_to_data_uri(
    current_image["path"],
    max_size=700
)

current_label = current_image["label"]

ai_prediction, ai_confidence = st.session_state.ai_prediction


# =========================================================
# TITLE / HEADER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        padding:10px 0 5px 0;
    ">
        <h1 style="
            font-size:42px;
            margin-bottom:0;
        ">
            ♻️ Waste Sorting Game
        </h1>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HANDLE DROP CALLBACK
# =========================================================

def handle_drop_change():
    pass

# =========================================================
# GAME COMPONENT
# =========================================================

game_component = st.components.v2.component(

    name="waste_sorting_game",

    html="""

    <div class="game-wrapper">

        <div class="top-bar">

            <div class="stat-card">
                <span class="stat-label">
                    SCORE
                </span>

                <span
                    id="score"
                    class="stat-value"
                >
                    0
                </span>
            </div>


            <div class="stat-card">

                <span class="stat-label">
                    QUESTION
                </span>

                <span
                    id="question"
                    class="stat-value"
                >
                    1 / 10
                </span>

            </div>

        </div>


        <div class="instruction">
            Drag the image to the correct trash bin
        </div>


        <div
            id="result-panel"
            class="result-panel hidden"
        >

            <!-- LEFT : CORRECT / WRONG -->

            <div
                id="feedback"
                class="feedback"
            ></div>


            <!-- RIGHT : AI -->

            <div
                id="ai-result"
                class="ai-result"
            >

                <div class="ai-header">

                    <div class="ai-title">
                        🤖 AI Prediction
                    </div>

                    <div
                        id="ai-prediction"
                        class="ai-prediction"
                    >
                        -
                    </div>

                </div>


                <div class="confidence-header">

                    <span>
                        Confidence
                    </span>

                    <strong id="ai-confidence">
                        0%
                    </strong>

                </div>


                <div class="confidence-bar">

                    <div
                        id="confidence-fill"
                        class="confidence-fill"
                    ></div>

                </div>

            </div>

        </div>

        <div
            id="playfield"
            class="playfield"
        >

            <img
                id="waste-image"
                class="waste-image"
                draggable="false"
            />


            <div
                id="organic-bin"
                class="bin organic-bin"
            >

                <img
                    class="bin-image"
                    id="organic-bin-image"
                />

                <div class="bin-label">
                    ORGANIC
                </div>

            </div>


            <div
                id="inorganic-bin"
                class="bin inorganic-bin"
            >

                <img
                    class="bin-image"
                    id="inorganic-bin-image"
                />

                <div class="bin-label">
                    INORGANIC
                </div>

            </div>

        </div>

    </div>

    <!-- ========================================= -->
    <!-- GAME OVER MODAL -->
    <!-- ========================================= -->

    <div
        id="game-over-modal"
        class="game-over-modal"
    >

        <div class="game-over-backdrop"></div>

        <div class="game-over-card">

            <div class="game-over-icon">
                🎉
            </div>

            <div class="game-over-title">
                GAME COMPLETE!
            </div>

            <div class="final-score-label">
                YOUR SCORE
            </div>

            <div
                id="final-score"
                class="final-score"
            >
                0 / 100
            </div>

            <button
                id="restart-game-button"
                class="restart-game-button"
            >
                🔄 Play Again
            </button>

        </div>

    </div>

    """,

    css="""

    * {
        box-sizing: border-box;
    }


    .game-wrapper {

        width:100%;
        max-width:950px;

        margin:auto;

        font-family:
            Inter,
            system-ui,
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }


    .top-bar {

        display:flex;

        justify-content:center;

        gap:18px;

        margin-bottom:12px;
    }


    .stat-card {

        min-width:150px;

        padding:10px 20px;

        border-radius:16px;

        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.95),
                rgba(240,240,240,0.95)
            );

        box-shadow:
            0 5px 18px rgba(0,0,0,0.08);

        text-align:center;
    }


    .stat-label {

        display:block;

        font-size:12px;

        font-weight:700;

        letter-spacing:1.5px;

        opacity:0.55;
    }


    .stat-value {

        display:block;

        margin-top:2px;

        font-size:25px;

        font-weight:800;
    }


    .instruction {

        text-align:center;

        font-size:16px;

        margin-bottom:8px;

        opacity:0.65;
    }


    .feedback {

        width:max-content;

        max-width:90%;

        margin:
            4px auto
            10px auto;

        padding:
            8px 20px;

        border-radius:999px;

        font-size:21px;

        font-weight:800;

        text-align:center;
    }


    .feedback.correct {

        background:#d9f7df;

        color:#18752c;
    }


    .feedback.wrong {

        background:#ffe0e0;

        color:#a51d1d;
    }


    .feedback.hidden {

        display:none;
    }


    .playfield {

        position:relative;

        width:100%;

        height:580px;

        overflow:hidden;

        border-radius:28px;

        background:
            radial-gradient(
                circle at 50% 15%,
                rgba(255,255,255,0.95),
                rgba(238,243,239,0.97)
            );

        border:
            1px solid rgba(0,0,0,0.06);

        box-shadow:
            0 10px 35px rgba(0,0,0,0.08);
    }


    .waste-image {

        position:absolute;

        left:50%;

        top:22%;

        width:170px;

        height:170px;

        object-fit:contain;

        transform:
            translate(-50%, -50%)
            scale(1);

        cursor:grab;

        user-select:none;

        -webkit-user-select:none;

        touch-action:none;

        z-index:10;

        filter:
            drop-shadow(
                0 15px 14px rgba(0,0,0,0.15)
            );

        transition:
            transform 0.18s ease,
            filter 0.18s ease;
    }


    .waste-image.dragging {

        cursor:grabbing;

        filter:
            drop-shadow(
                0 22px 18px rgba(0,0,0,0.22)
            );
    }


    .bin {

        position:absolute;

        bottom:28px;

        width:240px;

        height:240px;

        display:flex;

        flex-direction:column;

        align-items:center;

        justify-content:flex-end;

        padding-bottom:5px;

        border-radius:28px;

        transition:
            transform 0.18s ease,
            filter 0.18s ease,
            box-shadow 0.18s ease;
    }


    .organic-bin {

        left:8%;
    }


    .inorganic-bin {

        right:8%;
    }


    .bin-image {

        width:190px;

        height:190px;

        object-fit:contain;

        pointer-events:none;

        filter:
            drop-shadow(
                0 10px 10px rgba(0,0,0,0.14)
            );
    }


    .bin-label {

        margin-top:-4px;

        font-size:17px;

        font-weight:900;

        letter-spacing:1.5px;
    }


    .bin.near {

        transform:scale(1.04);

        box-shadow:
            0 0 0 5px rgba(66,153,80,0.14),
            0 15px 40px rgba(0,0,0,0.12);
    }


    .inorganic-bin.near {

        box-shadow:
            0 0 0 5px rgba(75,120,200,0.15),
            0 15px 40px rgba(0,0,0,0.12);
    }


    @media (max-width:700px) {

        .playfield {

            height:510px;
        }


        .waste-image {

            width:130px;

            height:130px;
        }


        .bin {

            width:175px;

            height:190px;

            bottom:20px;
        }


        .bin-image {

            width:145px;

            height:145px;
        }


        .organic-bin {

            left:1%;
        }


        .inorganic-bin {

            right:1%;
        }


        .bin-label {

            font-size:14px;
        }

    }

    /* =========================================
    GAME OVER MODAL
    ========================================= */

    .game-over-modal {

        position:fixed;

        inset:0;

        z-index:9999;

        display:none;

        align-items:center;

        justify-content:center;

    }


    /* Saat game selesai */

    .game-over-modal.show {

        display:flex;

    }


    /* =========================================
    BACKDROP
    ========================================= */

    .game-over-backdrop {

        position:absolute;

        inset:0;

        background:
            rgba(0,0,0,0.55);

        backdrop-filter:
            blur(6px);

        -webkit-backdrop-filter:
            blur(6px);

    }


    /* =========================================
    MODAL CARD
    ========================================= */

    .game-over-card {

        position:relative;

        z-index:2;

        width:min(
            430px,
            calc(100% - 40px)
        );

        padding:35px 30px 30px;

        border-radius:28px;

        text-align:center;

        background:
            linear-gradient(
                145deg,
                #ffffff,
                #f2f8f3
            );

        box-shadow:
            0 30px 80px
            rgba(0,0,0,0.30);

        animation:
            modalPop
            0.35s
            ease-out;

    }


    /* =========================================
    ANIMATION
    ========================================= */

    @keyframes modalPop {

        0% {

            opacity:0;

            transform:
                scale(0.75)
                translateY(20px);

        }

        100% {

            opacity:1;

            transform:
                scale(1)
                translateY(0);

        }

    }


    /* =========================================
    ICON
    ========================================= */

    .game-over-icon {

        font-size:64px;

        line-height:1;

        margin-bottom:12px;

    }


    /* =========================================
    TITLE
    ========================================= */

    .game-over-title {

        font-size:30px;

        font-weight:900;

        color:#18752c;

    }


    /* =========================================
    SCORE
    ========================================= */

    .final-score-label {

        margin-top:25px;

        font-size:12px;

        font-weight:800;

        letter-spacing:2px;

        color:#7c8780;

    }


    .final-score {

        margin-top:2px;

        font-size:52px;

        line-height:1.1;

        font-weight:900;

        color:#1d6f31;

    }

    /* =========================================
    BUTTON
    ========================================= */

    .restart-game-button {

        width:100%;

        margin-top:25px;

        padding:13px 20px;

        border:none;

        border-radius:14px;

        background:
            linear-gradient(
                135deg,
                #2f9e44,
                #18752c
            );

        color:white;

        font-size:16px;

        font-weight:800;

        cursor:pointer;

        box-shadow:
            0 7px 18px
            rgba(24,117,44,0.25);

        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease;
    }


    .restart-game-button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 10px 25px
            rgba(24,117,44,0.32);

    }


    .restart-game-button:active {

        transform:
            translateY(0);

    }


    /* =========================================
    RESULT PANEL
    ========================================= */

    .result-panel {

        max-width:900px;

        margin:15px auto 0;

        display:grid;

        grid-template-columns:
            1fr
            1fr;

        gap:20px;

        align-items:center;

        padding:20px 24px;

        border-radius:22px;

        background:
            linear-gradient(
                135deg,
                #ffffff,
                #f5f8f6
            );

        box-shadow:
            0 8px 25px rgba(0,0,0,0.07);
    }


    .result-panel.hidden {

        display:none;
    }


    /* =========================================
    FEEDBACK
    ========================================= */

    .feedback {

        width:100%;

        margin:0;

        padding:18px;

        border-radius:16px;

        font-size:22px;

        font-weight:900;

        text-align:center;
    }


    .feedback.correct {

        background:#d9f7df;

        color:#18752c;
    }


    .feedback.wrong {

        background:#ffe0e0;

        color:#a51d1d;
    }


    /* =========================================
    AI RESULT
    ========================================= */

    .ai-result {

        width:100%;
    }


    .ai-header {

        display:flex;

        justify-content:space-between;

        align-items:center;

        gap:12px;

        margin-bottom:12px;
    }


    .ai-title {

        font-size:16px;

        font-weight:800;

        color:#333;
    }


    .ai-prediction {

        padding:6px 13px;

        border-radius:999px;

        background:#dff4e3;

        color:#18752c;

        font-size:14px;

        font-weight:800;
    }


    /* =========================================
    CONFIDENCE
    ========================================= */

    .confidence-header {

        display:flex;

        justify-content:space-between;

        margin-bottom:7px;

        font-size:13px;

        color:#777;
    }


    .confidence-header strong {

        color:#18752c;

        font-weight:900;
    }


    .confidence-bar {

        width:100%;

        height:12px;

        background:#e5ebe6;

        border-radius:999px;

        overflow:hidden;
    }


    .confidence-fill {

        width:0%;

        height:100%;

        border-radius:999px;

        background:
            linear-gradient(
                90deg,
                #81c784,
                #2e7d32
            );

        transition:
            width 0.5s ease;
    }


    /* =========================================
    MOBILE
    ========================================= */

    @media (max-width:700px) {

        .result-panel {

            grid-template-columns:1fr;

            gap:14px;

            padding:16px;
        }

    }


    """,

    js="""

    export default function ({
        parentElement,
        data,
        setTriggerValue
    }) {

        const playfield =
            parentElement.querySelector("#playfield");

        const waste =
            parentElement.querySelector("#waste-image");

        const organicBin =
            parentElement.querySelector("#organic-bin");

        const inorganicBin =
            parentElement.querySelector("#inorganic-bin");

        const organicBinImage =
            parentElement.querySelector("#organic-bin-image");

        const inorganicBinImage =
            parentElement.querySelector("#inorganic-bin-image");

        const feedback =
            parentElement.querySelector("#feedback");

        const resultPanel =
            parentElement.querySelector(
                "#result-panel"
            );

        const score =
            parentElement.querySelector("#score");

        const question =
            parentElement.querySelector("#question");

        const gameOverModal =
            parentElement.querySelector(
                "#game-over-modal"
            );

        const finalScore =
            parentElement.querySelector(
                "#final-score"
            );

        const restartButton =
            parentElement.querySelector(
                "#restart-game-button"
            );

        const aiResult =
            parentElement.querySelector(
                "#ai-result"
            );

        const aiPrediction =
            parentElement.querySelector(
                "#ai-prediction"
            );

        const aiConfidence =
            parentElement.querySelector(
                "#ai-confidence"
            );

        const confidenceFill =
            parentElement.querySelector(
                "#confidence-fill"
            );


        // =================================================
        // PLAY AGAIN
        // =================================================

        if (
            restartButton &&
            !restartButton._initialized
        ) {

            restartButton._initialized = true;

            restartButton.addEventListener(
                "click",
                function(event) {

                    event.preventDefault();
                    event.stopPropagation();

                    setTriggerValue(
                        "play_again",
                        {
                            id:
                                Date.now() +
                                "_" +
                                Math.random()
                                    .toString(36)
                                    .slice(2)
                        }
                    );

                }
            );

        }




        // =================================================
        // UPDATE DATA
        // =================================================

        score.textContent =
            data?.score ?? 0;

        question.textContent =
            `${data?.question ?? 1} / ${data?.total_questions ?? 10}`;

        // AI RESULT
        if (
            data?.ai_prediction
        ) {

            aiPrediction.textContent =
                data.ai_prediction;


            const confidence =
                (data?.ai_confidence ?? 0) * 100;


            aiConfidence.textContent =
                `${confidence.toFixed(2)}%`;


            confidenceFill.style.width =
                `${confidence}%`;


            aiResult.style.display =
                "block";

        }
        else {

            aiResult.style.display =
                "none";

        }


        // =================================================
        // GAME OVER MODAL
        // =================================================

        if (data?.game_over) {

            const totalQuestions =
                data?.total_questions ?? 10;

            const scoreValue =
                data?.score ?? 0;

            const maxScore =
                totalQuestions * 10;

            const percentage =
                Math.round(
                    (scoreValue / maxScore) * 100
                );


            finalScore.textContent =
                `${scoreValue} / ${maxScore}`;


            gameOverModal.classList.add(
                "show"
            );

        }
        else {

            gameOverModal.classList.remove(
                "show"
            );

        }


        waste.src =
            data?.image_url ?? "";


        organicBinImage.src =
            data?.organic_bin_url ?? "";


        inorganicBinImage.src =
            data?.inorganic_bin_url ?? "";


        // Save latest component data.

        waste._componentData =
            data;


        // =================================================
        // RESET FOR NEW QUESTION
        // =================================================

        waste._dropSent = false;


        waste.style.left =
            "50%";

        waste.style.top =
            "22%";

        waste.style.transform =
            "translate(-50%, -50%) scale(1)";


        organicBin.classList.remove(
            "near"
        );

        inorganicBin.classList.remove(
            "near"
        );


        // =================================================
        // RESULT PANEL
        // =================================================

        if (data?.feedback) {

            resultPanel.classList.remove(
                "hidden"
            );

            feedback.classList.remove(
                "hidden"
            );

            feedback.textContent =
                data.feedback;

            feedback.classList.remove(
                "correct",
                "wrong"
            );

            if (
                data.feedback_type ===
                "correct"
            ) {

                feedback.classList.add(
                    "correct"
                );

            }
            else {

                feedback.classList.add(
                    "wrong"
                );

            }

        }
        else {

            resultPanel.classList.add(
                "hidden"
            );

            feedback.classList.add(
                "hidden"
            );

            feedback.textContent =
                "";

        }




        // =================================================
        // DRAG EVENTS
        // =================================================

        if (!waste._dragInitialized) {

            waste._dragInitialized =
                true;


            let dragging =
                false;


            let pointerOffsetX =
                0;


            let pointerOffsetY =
                0;


            // -------------------------------------------------
            // WASTE CENTER
            // -------------------------------------------------

            function getWasteCenter() {

                const rect =
                    waste.getBoundingClientRect();


                return {

                    x:
                        rect.left +
                        rect.width / 2,

                    y:
                        rect.top +
                        rect.height / 2

                };

            }


            // -------------------------------------------------
            // BIN CENTER
            // -------------------------------------------------

            function getBinCenter(bin) {

                const rect =
                    bin.getBoundingClientRect();


                return {

                    x:
                        rect.left +
                        rect.width / 2,

                    y:
                        rect.top +
                        rect.height / 2

                };

            }


            // -------------------------------------------------
            // DISTANCE
            // -------------------------------------------------

            function getDistance(a, b) {

                const dx =
                    a.x - b.x;

                const dy =
                    a.y - b.y;


                return Math.sqrt(
                    dx * dx +
                    dy * dy
                );

            }


            // -------------------------------------------------
            // PROXIMITY
            // -------------------------------------------------

            function updateProximity() {

                const wasteCenter =
                    getWasteCenter();


                const organicCenter =
                    getBinCenter(
                        organicBin
                    );


                const inorganicCenter =
                    getBinCenter(
                        inorganicBin
                    );


                const organicDistance =
                    getDistance(
                        wasteCenter,
                        organicCenter
                    );


                const inorganicDistance =
                    getDistance(
                        wasteCenter,
                        inorganicCenter
                    );


                const threshold =
                    190;


                const nearOrganic =
                    organicDistance <
                    threshold;


                const nearInorganic =
                    inorganicDistance <
                    threshold;


                organicBin.classList.toggle(
                    "near",
                    nearOrganic
                );


                inorganicBin.classList.toggle(
                    "near",
                    nearInorganic
                );


                const isNear =
                    nearOrganic ||
                    nearInorganic;


                if (isNear) {

                    waste.style.transform =
                        "translate(-50%, -50%) scale(0.60)";

                }
                else {

                    waste.style.transform =
                        "translate(-50%, -50%) scale(1)";

                }


                return {

                    nearOrganic,
                    nearInorganic

                };

            }


            // =================================================
            // POINTER DOWN
            // =================================================

            waste.addEventListener(
                "pointerdown",
                function(event) {

                    // Jangan izinkan drag setelah game selesai
                    if (waste._componentData?.game_over) {
                        return;
                    }

                    if (
                        waste._dropSent
                    ) {
                        return;
                    }

                    dragging =
                        true;


                    waste.classList.add(
                        "dragging"
                    );


                    waste.setPointerCapture(
                        event.pointerId
                    );


                    const wasteRect =
                        waste.getBoundingClientRect();


                    pointerOffsetX =
                        event.clientX -
                        (
                            wasteRect.left +
                            wasteRect.width / 2
                        );


                    pointerOffsetY =
                        event.clientY -
                        (
                            wasteRect.top +
                            wasteRect.height / 2
                        );

                }
            );


            // =================================================
            // POINTER MOVE
            // =================================================

            waste.addEventListener(
                "pointermove",
                function(event) {

                    if (!dragging) {
                        return;
                    }


                    const fieldRect =
                        playfield.getBoundingClientRect();


                    const x =
                        event.clientX -
                        fieldRect.left -
                        pointerOffsetX;


                    const y =
                        event.clientY -
                        fieldRect.top -
                        pointerOffsetY;


                    waste.style.left =
                        `${x}px`;


                    waste.style.top =
                        `${y}px`;


                    updateProximity();

                }
            );


            // =================================================
            // POINTER UP
            // =================================================

            waste.addEventListener(
                "pointerup",
                function(event) {

                    if (!dragging) {
                        return;
                    }


                    dragging =
                        false;


                    waste.classList.remove(
                        "dragging"
                    );


                    const proximity =
                        updateProximity();


                    let selectedBin =
                        null;


                    // ------------------------------------------------
                    // BOTH BINS ARE NEAR
                    // Select closest one.
                    // ------------------------------------------------

                    if (
                        proximity.nearOrganic &&
                        proximity.nearInorganic
                    ) {

                        const wasteCenter =
                            getWasteCenter();


                        const organicCenter =
                            getBinCenter(
                                organicBin
                            );


                        const inorganicCenter =
                            getBinCenter(
                                inorganicBin
                            );


                        const organicDistance =
                            getDistance(
                                wasteCenter,
                                organicCenter
                            );


                        const inorganicDistance =
                            getDistance(
                                wasteCenter,
                                inorganicCenter
                            );


                        selectedBin =
                            organicDistance <=
                            inorganicDistance

                                ? "Organic"

                                : "Inorganic";

                    }


                    // ------------------------------------------------
                    // ORGANIC
                    // ------------------------------------------------

                    else if (
                        proximity.nearOrganic
                    ) {

                        selectedBin =
                            "Organik";

                    }


                    // ------------------------------------------------
                    // INORGANIC
                    // ------------------------------------------------

                    else if (
                        proximity.nearInorganic
                    ) {

                        selectedBin =
                            "Anorganik";

                    }


                    // ------------------------------------------------
                    // NOT CLOSE ENOUGH
                    // ------------------------------------------------

                    if (!selectedBin) {

                        waste.style.left =
                            "50%";


                        waste.style.top =
                            "22%";


                        waste.style.transform =
                            "translate(-50%, -50%) scale(1)";


                        return;

                    }


                    // ------------------------------------------------
                    // PREVENT DOUBLE DROP
                    // ------------------------------------------------

                    if (
                        waste._dropSent
                    ) {

                        return;

                    }


                    waste._dropSent =
                        true;


                    organicBin.classList.remove(
                        "near"
                    );


                    inorganicBin.classList.remove(
                        "near"
                    );


                    const latestData =
                        waste._componentData;


                    // ------------------------------------------------
                    // UNIQUE EVENT ID
                    // ------------------------------------------------

                    const dropId =
                        Date.now() +
                        "_" +
                        Math.random()
                            .toString(36)
                            .slice(2);


                    // ------------------------------------------------
                    // SEND EVENT TO STREAMLIT
                    // ------------------------------------------------

                    setTriggerValue(
                        "drop",
                        {

                            id:
                                dropId,

                            bin:
                                selectedBin,

                            image_id:
                                latestData.image_id

                        }
                    );

                }
            );


            // =================================================
            // POINTER CANCEL
            // =================================================

            waste.addEventListener(
                "pointercancel",
                function() {

                    dragging =
                        false;


                    waste.classList.remove(
                        "dragging"
                    );

                }
            );

        }

    }

    """
)

# =========================================================
# MOUNT COMPONENT
# =========================================================

component_result = game_component(

    key=f"waste-game-{st.session_state.game_nonce}",

    data={

        "image_url":
            current_image_uri,

        "organic_bin_url":
            organic_bin_uri,

        "inorganic_bin_url":
            inorganic_bin_uri,

        "image_id":
            current_image["path"],

        "score":
            st.session_state.score,

        "question":
            st.session_state.question,

        "total_questions":
            TOTAL_QUESTIONS,

        "feedback":
            st.session_state.feedback,

        "feedback_type":
            st.session_state.feedback_type,

        "game_over":
            st.session_state.game_over,

        "ai_prediction":
            st.session_state.last_ai_prediction,

        "ai_confidence":
            st.session_state.last_ai_confidence,

    },

    on_drop_change=handle_drop_change
)

# =========================================================
# HANDLE DROP EVENT
# =========================================================

drop = getattr(
    component_result,
    "drop",
    None
)

if isinstance(drop, dict):

    drop_id = drop.get("id")

    # -----------------------------------------------------
    # Prevent duplicate processing
    # -----------------------------------------------------

    if (
        drop_id
        and
        drop_id !=
        st.session_state.last_drop_id
    ):

        st.session_state.last_drop_id = drop_id


        selected_bin = drop.get("bin")


        dropped_image_id = drop.get("image_id")


        # -------------------------------------------------
        # Make sure event belongs to current image
        # -------------------------------------------------

        if (
            dropped_image_id ==
            current_image["path"]
        ):

            # =============================================
            # SIMPAN DATA GAMBAR YANG BARU SAJA DIJAWAB
            # =============================================

            answered_image = current_image

            answered_label = current_label

            answered_ai_prediction, answered_ai_confidence = (
                st.session_state.ai_prediction
            )


            # =============================================
            # SIMPAN AI RESULT
            # =============================================

            st.session_state.last_answered_image = (
                answered_image
            )

            st.session_state.last_ai_prediction = (
                answered_ai_prediction
            )

            st.session_state.last_ai_confidence = (
                answered_ai_confidence
            )

            # ---------------------------------------------
            # CHECK ANSWER
            # ---------------------------------------------

            is_correct = (
                selected_bin ==
                current_label
            )


            if is_correct:

                st.session_state.score += 10

                st.session_state.feedback = (
                    "✅ CORRECT! +10 POINTS"
                )

                st.session_state.feedback_type = (
                    "correct"
                )

            else:

                st.session_state.feedback = (
                    "❌ WRONG! +0 POINTS"
                )

                st.session_state.feedback_type = (
                    "wrong"
                )


            # =============================================
            # NEXT QUESTION
            # =============================================

            if (
                st.session_state.question
                >= TOTAL_QUESTIONS
            ):

                st.session_state.game_over = True


            else:

                st.session_state.question += 1


                next_image = choose_random_image(
                        exclude_path=
                            current_image["path"]
                    )


                st.session_state.current_image = next_image


                st.session_state.ai_prediction = predict_image(
                        next_image["path"]
                    )


            # =============================================
            # FORCE STREAMLIT RERUN
            # =============================================

            st.rerun()

# =========================================================
# HANDLE PLAY AGAIN EVENT
# =========================================================

play_again = getattr(
    component_result,
    "play_again",
    None
)

if isinstance(play_again, dict):

    play_again_id = play_again.get("id")

    if (
        play_again_id
        and
        play_again_id !=
        st.session_state.get("last_play_again_id")
    ):

        st.session_state.last_play_again_id = play_again_id

        # RESET GAME
        st.session_state.score = 0
        st.session_state.question = 1

        st.session_state.feedback = ""
        st.session_state.feedback_type = ""

        st.session_state.last_drop_id = None

        st.session_state.game_over = False

        # Reset AI result sebelumnya
        st.session_state.last_answered_image = None
        st.session_state.last_ai_prediction = None
        st.session_state.last_ai_confidence = None

        # Pilih gambar baru
        selected = choose_random_image()

        st.session_state.current_image = selected

        st.session_state.ai_prediction = predict_image(
            selected["path"]
        )

        # Buat instance JS baru
        st.session_state.game_nonce += 1

        st.rerun()


# =========================================================
# GAME OVER
# =========================================================

if st.session_state.game_over:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            margin-top:18px;
            padding:18px;
            border-radius:20px;
            background:rgba(255,255,255,0.8);
            box-shadow:0 8px 25px rgba(0,0,0,0.08);
        ">

            <div style="
                font-size:18px;
                opacity:0.65;
            ">
                GAME COMPLETE
            </div>

            <div style="
                font-size:40px;
                font-weight:900;
                margin-top:5px;
            ">
                🎉 {st.session_state.score} POINTS
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )
