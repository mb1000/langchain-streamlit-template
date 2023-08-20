python -m venv env
source env/bin/activate
env/bin/pip install -U pip
env/bin/pip install -r requirements.txt
export OPENAI_API_KEY='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'

streamlit run main.py --server.enableCORS false --server.enableXsrfProtection false

source env/bin/activate
export OPENAI_API_KEY='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'
streamlit run main6.py --server.enableCORS false --server.enableXsrfProtection false
# Merke dir: Auf dem Tisch steht ein Teller.
# Was steht auf dem Tisch?

# pip freeze > ./requirements.txt