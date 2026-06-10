"""Default configuration constants.  All values are overridable via CLI flags."""

# ---------------------------------------------------------------------------
# MERT model
# ---------------------------------------------------------------------------
MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SR = 24_000          # MERT expects 24 kHz mono
MERT_FRAME_RATE = 75        # MERT outputs ~75 frames per second
MERT_LAYER = 7              # transformer layer to extract (0=CNN, 1-12=transformer)

# ---------------------------------------------------------------------------
# Preprocessing: dense reference embeddings (offline, configurable via CLI)
# ---------------------------------------------------------------------------
LOOKBACK_SEC = 2.0          # sliding window lookback for mean-pooling
STRIDE_SEC = 0.020          # 20ms stride → dense reference sequence

# ---------------------------------------------------------------------------
# Live alignment
# ---------------------------------------------------------------------------
CHUNK_SEC = 0.200           # audio chunk size for live embedding updates
SILENCE_RMS_DBFS = -50.0    # skip alignment when lookback RMS is below this
                            # (mic open but song not playing: ambient noise
                            # still DTW-matches the song's quietest section)

# ---------------------------------------------------------------------------
# Subsequence DTW
# ---------------------------------------------------------------------------
DTW_LIVE_SEC = 6.0          # live buffer capacity fed to DTW comparison
DTW_MIN_LIVE_SEC = 4.0      # minimum buffer fill before DTW runs (warm-up gate)
DTW_SEARCH_SEC = 45.0       # forward search window in reference audio
DTW_BAND_RATIO = 0.1        # Sakoe-Chiba band as fraction of query length

# ---------------------------------------------------------------------------
# Initial lock & jump guard (repeat-ambiguity defenses)
# ---------------------------------------------------------------------------
INIT_TOP_K = 5              # cosine candidates DTW-refined during initial lock
INIT_CAND_SEP_SEC = 8.0     # min separation between those candidates
INIT_CONSISTENT_FRAMES = 3  # consecutive confident frames required to lock
INIT_AGREE_SEC = 3.0        # ...all within this span of each other
JUMP_GUARD_SEC = 5.0        # forward jumps larger than this need confirmation
JUMP_CONFIRM_FRAMES = 3     # consecutive agreeing frames to accept a big jump
JUMP_AGREE_SEC = 2.5        # agreement tolerance for those frames

# ---------------------------------------------------------------------------
# HMM predictor
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.55  # minimum DTW confidence to feed HMM as hard obs.
HMM_OBS_SIGMA_RATIO = 0.3    # emission sigma = slide_duration * this ratio
HMM_MIN_SIGMA_SEC = 0.5      # floor for emission sigma (handles short slides)

# ---------------------------------------------------------------------------
# Trigger scheduler
# ---------------------------------------------------------------------------
TRIGGER_BUFFER_MS = 200     # fire REST call this many ms before slide boundary
TRIGGER_CONFIDENCE_MIN = 0.6  # minimum HMM trigger confidence

# ---------------------------------------------------------------------------
# REST output
# ---------------------------------------------------------------------------
REST_URL = "http://localhost:5000/slide"
REST_TIMEOUT_SEC = 2.0
