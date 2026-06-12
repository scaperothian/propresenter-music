"""Default configuration constants.  All values are overridable via CLI flags."""

# ---------------------------------------------------------------------------
# MERT model
# ---------------------------------------------------------------------------
MODEL_ID = "m-a-p/MERT-v1-95M"
TARGET_SR = 24_000          # MERT expects 24 kHz mono
MERT_FRAME_RATE = 75        # MERT outputs ~75 frames per second
MERT_LAYER = 7              # transformer layer to extract (0=CNN, 1-12=transformer)
MERT_FP16 = True            # run MERT in float16 (faster on MPS).  Reference
                            # cache and live MUST use the same precision —
                            # rebuild caches after flipping this.

# ---------------------------------------------------------------------------
# Preprocessing: dense reference embeddings (offline, configurable via CLI)
# ---------------------------------------------------------------------------
LOOKBACK_SEC = 2.0          # sliding window lookback for mean-pooling
STRIDE_SEC = 0.10           # reference window stride.  0.1s validated equal
                            # to 0.05s on Drive + Your Way Is Better (all
                            # offsets: same fires to the ms, tracking 0.20s)
                            # at half the preprocessing cost and ~25% lower
                            # live matching latency.
EMBED_BATCH_SIZE = 16       # windows per MERT forward during preprocessing.
                            # Larger batches are bit-identical but NOT faster
                            # on MPS (GPU already saturated at 16); may help
                            # on CUDA.

# ---------------------------------------------------------------------------
# Live alignment
# ---------------------------------------------------------------------------
CHUNK_SEC = 0.200           # audio chunk size for live embedding updates
SILENCE_RMS_DBFS = -50.0    # skip alignment when lookback RMS is below this
                            # (mic open but song not playing: ambient noise
                            # still DTW-matches the song's quietest section)

# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
MATCHER = "rigid"           # "rigid" — fixed 1:1 time mapping (linear playback
                            #           of the reference recording; immune to
                            #           DTW's warp-absorbs-mismatch lag).
                            # "dtw"   — subsequence DTW (handles tempo warps;
                            #           live-band scenario).
                            # A/B on EQ'd mic-proxy audio: rigid 14/14 fires
                            # at -400ms +-5ms, tracking 0.20s; DTW 9/14 with
                            # +0.2..+2.3s drift.

# ---------------------------------------------------------------------------
# Subsequence DTW
# ---------------------------------------------------------------------------
DTW_LIVE_SEC = 6.0          # live buffer capacity fed to DTW comparison
DTW_MIN_LIVE_SEC = 4.0      # minimum buffer fill before DTW runs (warm-up gate)
DTW_SEARCH_SEC = 45.0       # forward search window in reference audio
DTW_BAND_RATIO = 0.1        # Sakoe-Chiba band as fraction of query length

# ---------------------------------------------------------------------------
# Live-mean adaptation (cross-acoustic robustness)
# ---------------------------------------------------------------------------
LIVE_MEAN_ADAPT_SEC = 20.0  # blend from the cache's song mean to the live
                            # stream's own running mean over this much salient
                            # audio.  PA/room/mic coloration shifts every live
                            # embedding by a common offset; subtracting the
                            # live mean cancels it (the outro otherwise
                            # becomes everything's nearest neighbour).

# ---------------------------------------------------------------------------
# Initial lock & jump guard (repeat-ambiguity defenses)
# ---------------------------------------------------------------------------
INIT_TOP_K = 5              # cosine candidates DTW-refined during initial lock
INIT_CAND_SEP_SEC = 8.0     # min separation between those candidates
INIT_CONSISTENT_FRAMES = 3  # consecutive confident frames required to lock
INIT_AGREE_SEC = 3.0        # ...all within this span of each other
INIT_BUFFER_SEC = 16.0      # pre-lock DTW query capacity — longer than one
                            # slide (~14s avg) so the query spans a section
                            # transition; identical repeats (chorus pairs) are
                            # indistinguishable without one.  Costs no extra
                            # MERT compute, only cheap DTW.
INIT_MIN_COST_MARGIN = 0.05  # refuse to lock while best-vs-runner-up DTW cost
                             # margin is below this (tie = ambiguous repeat).
                             # Live wrong-lock ties measured 0.00-0.047
                             # (/tmp/ppsync.jsonl, chorus join at 62s);
                             # correct studio locks measure ~0.23.
JUMP_GUARD_SEC = 5.0        # forward jumps larger than this need confirmation
JUMP_CONFIRM_FRAMES = 3     # consecutive agreeing frames to accept a big jump
JUMP_AGREE_SEC = 2.5        # agreement tolerance for those frames
JUMP_MIN_COST_MARGIN = 0.05  # ...and the jump target's DTW cost must beat a
                             # local re-alignment near the current anchor by
                             # this margin — a stable wrong match agrees with
                             # itself, so stability alone cannot accept jumps
                             # (colored-audio trace: runaway 25s -> 102 -> 219)

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
