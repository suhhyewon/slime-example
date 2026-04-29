#!/bin/bash
# Local-friendly version of install_dependency.sh.
# Differences from the original:
#   - cp lex_lookup target moved to ${SCRATCH_DIR}/bin (no root needed)
#   - flite is built under ${SCRATCH_DIR}/build instead of $HOME (saves /nethome quota)
#   - flite's `make install` is replaced with a no-op (we only need lex_lookup)
#   - spacy/nltk data installs unchanged (they go to user-writable paths by default)
#
# This script installs *only* the slime-example IF-verifier deps. The SLIME
# framework, Megatron-LM, sglang, flash-attn, apex, etc. are installed
# separately — see /var/tmp/hsuh45-slime/yxli2123-slime/build_conda.sh and the
# README in that repo.
#
# Run inside an activated conda env (e.g. `micromamba activate slime`).

set -ex

SCRATCH_DIR=${SCRATCH_DIR:-/var/tmp/hsuh45-slime}
mkdir -p "${SCRATCH_DIR}/bin" "${SCRATCH_DIR}/build"

# ===================== Python deps =====================
pip install --index-url https://pypi.org/simple \
  openai pronouncing epitran langdetect spacy beautifulsoup4 nltk toml datasets

# aws_bedrock_token_generator is imported by multi_if_reward.py but only used if
# the colleague's AWS judge endpoint is targeted. Safe to install for parity.
pip install --index-url https://pypi.org/simple aws_bedrock_token_generator || \
  echo "WARNING: aws_bedrock_token_generator install failed; continuing (only needed for AWS Bedrock judge)."

# ===================== flite (for lex_lookup) =====================
cd "${SCRATCH_DIR}/build"
if [ ! -d flite ]; then
  git clone https://github.com/festvox/flite.git
fi
cd flite
if [ ! -f testsuite/lex_lookup ]; then
  sh configure
  make
  cd testsuite
  make lex_lookup
  cd ..
fi
cp testsuite/lex_lookup "${SCRATCH_DIR}/bin/lex_lookup"
echo "Installed lex_lookup to ${SCRATCH_DIR}/bin/lex_lookup"
echo "Make sure ${SCRATCH_DIR}/bin is on PATH before running training."

# ===================== spaCy / NLTK data =====================
python -m spacy download en_core_web_sm
python -m nltk.downloader averaged_perceptron_tagger_eng
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
python -m nltk.downloader cmudict
python -m nltk.downloader wordnet

echo ""
echo "===================================================================="
echo "Done. To verify lex_lookup is on PATH:"
echo "  export PATH=${SCRATCH_DIR}/bin:\$PATH"
echo "  which lex_lookup"
echo "===================================================================="
