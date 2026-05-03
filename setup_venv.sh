#!/bin/bash
VENV_NAME="$(basename "$(pwd)")_venv"
python3 -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "No requirements.txt found"
fi

echo "Virtual environment '$VENV_NAME' created and activated"

exec "$SHELL"