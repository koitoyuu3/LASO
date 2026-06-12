from flask import Flask
import logging

from ollama_agent_api import ollama_agent_bp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.register_blueprint(ollama_agent_bp)



@app.get("/health")
def health():
    return {"success": True, "message": "ollama agent app ready"}

if __name__ == '__main__':
    logger.info("starting ollama agent flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)
